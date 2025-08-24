#!/usr/bin/env python3

"""
Simple Ollama-compatible FastAPI server for running custom models locally
Supports both transformers models and GGUF models
"""

import os
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class ModelInfo(BaseModel):
    name: str
    size: int
    digest: str
    modified_at: str


class TagsResponse(BaseModel):
    models: List[ModelInfo]


class LocalModelServer:
    def __init__(self, model_path: str, model_name: str = "custom-model", 
                 max_seq_length: int = 2048, temperature: float = 0.3, 
                 load_in_4bit: bool = True):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_seq_length = max_seq_length
        self.temperature = temperature
        self.load_in_4bit = load_in_4bit
        self.is_gguf = self._check_if_gguf()
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Ollama-Compatible Local Model Server")
        self._setup_routes()
        
    def _check_if_gguf(self) -> bool:
        """Check if the model path is a GGUF file"""
        if os.path.isfile(self.model_path) and self.model_path.endswith('.gguf'):
            return True
        if os.path.isdir(self.model_path):
            gguf_files = list(Path(self.model_path).glob("*.gguf"))
            return len(gguf_files) > 0
        return False

    def _load_gguf_model(self):
        """Load GGUF model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            if os.path.isfile(self.model_path):
                model_file = self.model_path
            else:
                gguf_files = list(Path(self.model_path).glob("*.gguf"))
                if not gguf_files:
                    raise ValueError("No GGUF files found in directory")
                model_file = str(gguf_files[0])
            
            print(f"Loading GGUF model: {model_file}")
            self.model = Llama(
                model_path=model_file,
                n_ctx=self.max_seq_length,
                n_threads=os.cpu_count(),
                verbose=False
            )
            print("GGUF model loaded successfully!")
            
        except ImportError:
            print("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise
        except Exception as e:
            print(f"Failed to load GGUF model: {e}")
            raise

    def _load_transformers_model(self):
        """Load transformers model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"Loading transformers model: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=False
            )
            
            # Load model
            model_kwargs = {
                "trust_remote_code": False,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            if self.load_in_4bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    print("Using 4-bit quantization")
                except ImportError:
                    print("BitsAndBytesConfig not available, loading without quantization")
            
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Transformers model loaded successfully!")
            
        except ImportError as e:
            print(f"Required libraries not installed: {e}")
            print("Install with: pip install torch transformers")
            raise
        except Exception as e:
            print(f"Failed to load transformers model: {e}")
            raise
    
    def load_model(self):
        """Load the model based on type"""
        print(f"Model type: {'GGUF' if self.is_gguf else 'Transformers'}")
        
        if self.is_gguf:
            self._load_gguf_model()
        else:
            self._load_transformers_model()
    
    def _generate_gguf(self, prompt: str, stream: bool = False, **kwargs):
        """Generate text using GGUF model"""
        max_tokens = kwargs.get('max_tokens', 512)
        temperature = kwargs.get('temperature', self.temperature)
        
        if stream:
            # Streaming response
            def stream_generator():
                for output in self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stop=["<|im_end|>", "<|endoftext|>"],
                    echo=False
                ):
                    if output['choices'][0]['text']:
                        chunk = {
                            "model": self.model_name,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            "response": output['choices'][0]['text'],
                            "done": False
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                # Final chunk
                final_chunk = {
                    "model": self.model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "response": "",
                    "done": True
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
            
            return stream_generator()
        else:
            # Non-streaming response
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"],
                echo=False
            )
            return output['choices'][0]['text']
    
    def _generate_transformers(self, prompt: str, stream: bool = False, **kwargs):
        """Generate text using transformers model"""
        import torch
        
        max_tokens = kwargs.get('max_tokens', 512)
        temperature = kwargs.get('temperature', self.temperature)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                              max_length=self.max_seq_length)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if stream:
            # Simple streaming (generate word by word)
            def stream_generator():
                generated_text = ""
                
                for i in range(max_tokens):
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1,
                            **generation_kwargs
                        )
                    
                    # Decode new token
                    new_token = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    if new_token and new_token not in ["<|im_end|>", "<|endoftext|>"]:
                        generated_text += new_token
                        chunk = {
                            "model": self.model_name,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            "response": new_token,
                            "done": False
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # Update inputs for next iteration
                    inputs['input_ids'] = outputs
                    if 'attention_mask' in inputs:
                        inputs['attention_mask'] = torch.ones_like(outputs)
                    
                    # Check for stop condition
                    if self.tokenizer.eos_token_id in outputs[0][-5:]:
                        break
                
                # Final chunk
                final_chunk = {
                    "model": self.model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "response": "",
                    "done": True
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
            
            return stream_generator()
        else:
            # Non-streaming response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return generated_text

    def generate(self, prompt: str, stream: bool = False, **kwargs):
        """Generate text using the appropriate model"""
        if self.is_gguf:
            return self._generate_gguf(prompt, stream, **kwargs)
        else:
            return self._generate_transformers(prompt, stream, **kwargs)

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {"message": "Ollama-compatible local model server", "model": self.model_name}

        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "model_name": self.model_name,
                "model_type": "GGUF" if self.is_gguf else "Transformers"
            }

        @self.app.get("/api/tags")
        async def list_models():
            """Ollama-compatible model listing"""
            model_info = ModelInfo(
                name=self.model_name,
                size=0,  # Could calculate actual size
                digest="local",
                modified_at=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            )
            return TagsResponse(models=[model_info])

        @self.app.post("/api/generate")
        async def generate_text(request: GenerateRequest):
            """Ollama-compatible generate endpoint"""
            if self.model is None:
                raise HTTPException(status_code=500, detail="Model not loaded")

            try:
                # Extract options
                options = request.options or {}
                max_tokens = options.get('num_predict', 512)
                temperature = options.get('temperature', self.temperature)

                start_time = time.time()

                if request.stream:
                    # Streaming response
                    generator = self.generate(
                        request.prompt, 
                        stream=True,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return StreamingResponse(
                        generator,
                        media_type="text/plain",
                        headers={"Content-Type": "text/plain; charset=utf-8"}
                    )
                else:
                    # Non-streaming response
                    response_text = self.generate(
                        request.prompt,
                        stream=False,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )

                    total_duration = int((time.time() - start_time) * 1e9)  # nanoseconds

                    return GenerateResponse(
                        model=self.model_name,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        response=response_text,
                        done=True,
                        total_duration=total_duration,
                        eval_count=len(response_text.split()),
                        eval_duration=total_duration // 2  # rough estimate
                    )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/chat")
        async def chat(request: ChatRequest):
            """Ollama-compatible chat endpoint"""
            if self.model is None:
                raise HTTPException(status_code=500, detail="Model not loaded")

            try:
                # Convert messages to prompt
                prompt_parts = []
                for msg in request.messages:
                    if msg.role == "system":
                        prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
                    elif msg.role == "user":
                        prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
                    elif msg.role == "assistant":
                        prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

                prompt_parts.append("<|im_start|>assistant\n")
                prompt = "\n".join(prompt_parts)

                # Use generate endpoint logic
                generate_request = GenerateRequest(
                    model=request.model,
                    prompt=prompt,
                    stream=request.stream,
                    options=request.options
                )

                return await generate_text(generate_request)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/shutdown")
        def shutdown():
            os.kill(os.getpid(), signal.SIGTERM)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Ollama-compatible local model server")
    parser.add_argument("model_path", help="Path to model directory or GGUF file")
    parser.add_argument("--model-name", default="custom-model", help="Model name identifier")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.2, help="Default temperature")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return 1

    # Create server
    server = LocalModelServer(
        model_path=args.model_path,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        temperature=args.temperature,
        load_in_4bit=not args.no_4bit
    )

    print(f"Loading model from: {args.model_path}")
    try:
        server.load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model: {args.model_name}")
    print(f"Type: {'GGUF' if server.is_gguf else 'Transformers'}")
    print("\nEndpoints:")
    print(f"  - Health: http://{args.host}:{args.port}/health")
    print(f"  - Generate: http://{args.host}:{args.port}/api/generate")
    print(f"  - Chat: http://{args.host}:{args.port}/api/chat")
    print(f"  - Models: http://{args.host}:{args.port}/api/tags")

    print("\nExample usage:")
    print(f"""curl -X POST http://localhost:{args.port}/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{args.model_name}", "prompt": "Write a Python function"}}'""")

    try:
        uvicorn.run(
            server.app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped")
        return 0
    except Exception as e:
        print(f"Server error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
