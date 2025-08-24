#!/usr/bin/env python3

# file finetuner.py

"""
Fine-tuning script for Qwen2.5 models using Unsloth and LoRA
Optimized for coding style and convention learning from datasets
"""

from unsloth import FastLanguageModel

import os
import json
from typing import List, Dict, Any
from pathlib import Path
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import argparse
from dataset_generator.qwen_dataset_generator import QwenDatasetGenerator
from dataset_generator.utils import setup_logging
from dataset_generator.utils import setup_logging



class Qwen25FineTuner:
    def __init__(
        self,
        model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length: int = 2048,
        dtype: torch.dtype = None,
        load_in_4bit: bool = True,
    ):
        """
        Initialize the Qwen2.5 fine-tuner
        
        Args:
            model_name: Hugging Face model name or path
            max_seq_length: Maximum sequence length for training
            dtype: Data type for model weights (None for auto)
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer with Unsloth optimizations"""
        print(f"Loading model: {self.model_name}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            # Trust remote code for Qwen models
            trust_remote_code=False,
        )
        
        # Setup LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # LoRA rank
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
            random_state=3407,
            use_rslora=False,  # Rank stabilized LoRA
            loftq_config=None,  # LoftQ
        )

        print("Model loaded successfully with LoRA adapters!")

    def format_dataset(self, raw_data):
        # Convert to Hugging Face dataset format
        formatted_data = []
        for item in raw_data:
            # Format for chat template
            messages = item["messages"]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            formatted_data.append({"text": text})

        dataset = Dataset.from_list(formatted_data)
        print(f"Dataset loaded with {len(dataset)} examples")
        return dataset

    def load_dataset(self, dataset_path: str) -> Dataset:
        """
        Load and process the training dataset
        Args:
            dataset_path: Path to the JSONL dataset file
        Returns:
            Processed Hugging Face Dataset
        """
        print(f"Loading dataset from: {dataset_path}")
        
        # Load JSONL data
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        return self.format_dataset(data)

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./qwen25-finetuned",
        num_train_epochs: int = 3,
        learning_rate: float = 2e-4,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: int = 60,
        logging_steps: int = 1,
        save_steps: int = 20,

        **kwargs
    ):
        """
        Fine-tune the model using SFT (Supervised Fine-Tuning)
        
        Args:
            dataset: Training dataset
            output_dir: Directory to save the fine-tuned model
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            logging_steps: Logging frequency
            save_steps: Model saving frequency
        """
        print("Starting fine-tuning process...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            save_steps=save_steps,
            save_total_limit=3,
            report_to="none",
            remove_unused_columns=False,
            **kwargs
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences
            args=training_args,
        )
        
        # Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        
        # Start training
        trainer_stats = trainer.train()
        
        # Memory stats after training
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        
        return trainer_stats

    def save_model_variants(self, output_dir: str, model_name: str = "qwen25-coding-style"):
        """
        Save the model in different formats

        Args:
            output_dir: Base output directory
            model_name: Name for the saved model
        """
        print("Saving model variants...")

        # Save merged model (16-bit)
        merged_model_path = os.path.join(output_dir, f"{model_name}-merged-16bit")
        self.model.save_pretrained_merged(
            merged_model_path,
            self.tokenizer,
            save_method="merged_16bit"
        )
        print(f"16-bit merged model saved to: {merged_model_path}")

        # Save merged model (4-bit)
        merged_model_4bit_path = os.path.join(output_dir, f"{model_name}-merged-4bit")
        self.model.save_pretrained_merged(
            merged_model_4bit_path,
            self.tokenizer,
            save_method="merged_4bit_forced"
        )
        print(f"4-bit merged model saved to: {merged_model_4bit_path}")

        # Save LoRA adapters only
        lora_path = os.path.join(output_dir, f"{model_name}-lora")
        self.model.save_pretrained(lora_path)
        self.tokenizer.save_pretrained(lora_path)
        print(f"LoRA adapters saved to: {lora_path}")

    def export_to_gguf(self, output_dir: str, model_name: str = "qwen25-coding-style"):
        """
        Export the fine-tuned model to GGUF format

        Args:
            output_dir: Output directory
            model_name: Model name for GGUF file
        """
        try:
            print("Exporting to GGUF format...")
            gguf_path = os.path.join(output_dir, f"{model_name}.gguf")

            self.model.save_pretrained_gguf(
                gguf_path,
                self.tokenizer,
                quantization_method="q4_k_m"  # You can choose different quantization methods
            )
            print(f"GGUF model saved to: {gguf_path}")

        except Exception as e:
            print(f"GGUF export failed: {e}")
            print("Note: GGUF export requires additional dependencies. Install with:")
            print("pip install llama-cpp-python")


def generate_dataset(repo_path: str, max_examples: int, llm_server: str, llm_name: str):
    """
    Generates Dataset using LLM
    """
    if not os.path.exists(repo_path):
        print(f"Error: Repository path '{repo_path}' does not exist")
        return 1

    # Initialize generator
    generator = QwenDatasetGenerator(
        repo_path=repo_path,
        max_examples=max_examples,
        llm_server_url=llm_server,
        llm_model=llm_name
    )

    print("Generating Qwen2.5 + QLoRA optimized dataset...")
    print(f"Using LLM at {llm_server} with model {llm_name} for prompt generation")

    dataset = generator.generate_qwen_dataset()

    if not dataset:
        print("No examples generated. Check if the repository contains supported code files.")
        return 1

    return dataset
    # generator.save_dataset(dataset, output_dir)


def main():
    

    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 with Unsloth and LoRA")
    parser.add_argument("--dataset_repo", type=str, required=True, help="Path to repository/codebase")
    parser.add_argument("--max_examples", type=int, default=3000, help="Maximum number of examples")
    parser.add_argument("--trainer_host", type=str, required=True, help="Url for trainer model server (Ollama or compatible)")
    parser.add_argument("--trainer_name", type=str, required=True, help="Name of trainer model")
    
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./models/finetuned", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=60, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--save_variants", action="store_true", help="Save model in multiple formats")
    parser.add_argument("--export_gguf", action="store_true", help="Export to GGUF format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    # Check if dataset exists
    # if not os.path.exists(args.dataset):
    #     print(f"Error: Dataset file not found: {args.dataset}")
    #     return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize fine-tuner
    fine_tuner = Qwen25FineTuner(
        model_name=args.model,
        max_seq_length=args.max_seq_length
    )

    try:
        dataset_list = generate_dataset(
            repo_path=args.dataset_repo,
            max_examples=args.max_examples,
            llm_server=args.trainer_host,
            llm_name=args.trainer_name)

        # Load model
        fine_tuner.load_model()
        dataset = fine_tuner.format_dataset(dataset_list)

        # Load dataset
        # dataset = fine_tuner.load_dataset(args.dataset)
        # repo_path = "/home/noroot/Desktop/mnt/myfiles/koz_cli/repositories/full-stack-fastapi-codebase/"

        # Train the model
        trainer_stats = fine_tuner.train(
            dataset=dataset,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
        )
        
        print("Training completed successfully!")
        
        # Save model variants if requested
        if args.save_variants:
            fine_tuner.save_model_variants(args.output_dir)
        
        # Export to GGUF if requested
        if args.export_gguf:
            fine_tuner.export_to_gguf(args.output_dir)
            
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        raise

if __name__ == "__main__":
    main()
    # repo_path = "/home/noroot/Desktop/mnt/myfiles/koz_cli/repositories/full-stack-fastapi-codebase/"
    # generate_dataset(
    #     repo_path=repo_path,
    #     output_dir="./dataset/7b/qwen2.5_3b.json",
    #     max_examples=5,
    #     llm_server="http://localhost:8000",
    #     llm_name="qwen2.5-coder:7b")
