import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .parsers import PythonParser, JavaScriptParser, JavaParser, CppParser, GoParser, RustParser
from .llm_prompt_generator import LLMPromptGenerator
from .style_analyzer import StyleAnalyzer
from .utils import create_output_directory
# import random


@dataclass
class CodeExample:
    instruction: str
    input_context: str
    output: str
    metadata: Dict[str, Any]


class QwenDatasetGenerator:
    """
    Dataset generator optimized for Qwen2.5 + QLoRA fine-tuning for coding style adaptation.
    """

    def __init__(self, repo_path: str, max_examples: int = 5000, 
                 llm_server_url: Optional[str] = None, llm_model: Optional[str] = None):
        self.repo_path = Path(repo_path)
        self.max_examples = max_examples
        self.examples = []
        self.logger = logging.getLogger(__name__)

        # Initialize style analyzer
        self.style_analyzer = StyleAnalyzer()

        # Initialize LLM prompt generator if specified
        self.llm_generator = None
        if llm_server_url and llm_model:
            self.llm_generator = LLMPromptGenerator(llm_server_url, llm_model)
            if not self.llm_generator.test_connection():
                self.logger.warning("LLM server connection failed, falling back to template prompts")
                self.llm_generator = None

        # Initialize parsers
        self.parsers = {
            '.py': PythonParser(self.style_analyzer),
            '.js': JavaScriptParser(self.style_analyzer),
            '.ts': JavaScriptParser(self.style_analyzer),
            '.java': JavaParser(self.style_analyzer),
            '.cpp': CppParser(self.style_analyzer),
            '.c': CppParser(self.style_analyzer),
            '.h': CppParser(self.style_analyzer),
            '.hpp': CppParser(self.style_analyzer),
            '.go': GoParser(self.style_analyzer),
            '.rs': RustParser(self.style_analyzer),
        }
        
        # Qwen2.5 specific instruction templates
        self.instruction_templates = {
            'function_style': [
                "Write a function following this codebase's naming and documentation style",
                "Implement a function that matches the coding patterns in this repository",
                "Create a function using the established conventions from this project",
            ],
            'class_style': [
                "Design a class following this project's architectural patterns",
                "Implement a class that adheres to this codebase's style guidelines",
                "Create a class structure matching this repository's conventions",
            ],
            'method_style': [
                "Write a method that follows this class's established patterns",
                "Implement a method using the consistent style from this codebase",
                "Add a method that matches the existing code structure",
            ],
            'error_handling': [
                "Implement error handling following this project's patterns",
                "Add exception handling using the established conventions",
                "Write error handling code matching this codebase's style",
            ],
            'refactor_style': [
                "Refactor this code to match the repository's style guidelines",
                "Improve this code following the project's best practices",
                "Rewrite this function to align with the codebase conventions",
            ]
        }
    
    def collect_files(self) -> List[Path]:
        """Collect all code files from the repository."""
        files = []
        for ext in self.parsers.keys():
            files.extend(self.repo_path.rglob(f"*{ext}"))
        
        # Filter out common directories to ignore
        ignore_patterns = [
            'node_modules', '__pycache__', '.git', 'venv', 'env',
            'build', 'dist', '.pytest_cache', 'target', 'bin', 'obj',
            'vendor', '.next', 'coverage', '.nyc_output', 'migrations'
        ]
        
        filtered_files = []
        for file in files:
            if not any(pattern in str(file) for pattern in ignore_patterns):
                # Skip very large files (>100KB) and very small files (<100 bytes)
                try:
                    size = file.stat().st_size
                    if 100 <= size <= 100000:
                        filtered_files.append(file)
                except OSError:
                    continue
                
        return filtered_files
    
    def generate_qwen_dataset(self) -> List[Dict[str, Any]]:
        """Generate dataset optimized for Qwen2.5 + QLoRA fine-tuning."""
        files = self.collect_files()
        self.logger.info(f"Found {len(files)} code files")
        
        if not files:
            self.logger.error("No suitable files found in repository")
            return []
        
        # Analyze style patterns first
        self.logger.info("Analyzing codebase style patterns...")
        self.style_analyzer.analyze_repository(files[:50])  # Analyze first 50 files for patterns
        
        all_examples = []
        processed_files = 0
        
        for file_path in files:
            if len(all_examples) >= self.max_examples:
                break
                
            try:
                self.logger.debug(f"Processing: {file_path}")
                
                ext = file_path.suffix.lower()
                if ext in self.parsers:
                    parser = self.parsers[ext]
                    examples = parser.parse_file(file_path)
                    
                    # Enhance examples with LLM-generated prompts if available
                    if self.llm_generator:
                        examples = self._enhance_with_llm_prompts(examples)
                    
                    all_examples.extend(examples)
                    processed_files += 1
                    
                    if processed_files % 10 == 0:
                        self.logger.info(f"Processed {processed_files} files, generated {len(all_examples)} examples")
                        
            except Exception as e:
                self.logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        self.logger.info(f"Initial extraction: {len(all_examples)} examples from {processed_files} files")
        
        # Process and filter examples
        all_examples = self._deduplicate_examples(all_examples)
        all_examples = self._filter_quality_examples(all_examples)
        
        # Convert to Qwen2.5 chat format
        dataset = self._convert_to_qwen_format(all_examples[:self.max_examples])
        
        self.logger.info(f"Final dataset: {len(dataset)} training examples for Qwen2.5 + QLoRA")
        return dataset
    
    def _enhance_with_llm_prompts(self, examples: List[CodeExample]) -> List[CodeExample]:
        """Enhance examples with LLM-generated prompts."""
        if not self.llm_generator:
            return examples
        
        enhanced_examples = []
        
        for example in examples:
            try:
                # Generate more natural prompt using LLM
                llm_prompt = self.llm_generator.generate_prompt(
                    example.output,
                    fallback_instruction=example.instruction
                )
                
                # Create enhanced example
                enhanced_example = CodeExample(
                    instruction=llm_prompt,
                    input_context=example.input_context,
                    output=example.output,
                    metadata={
                        **example.metadata,
                        'llm_generated_prompt': True,
                        'original_instruction': example.instruction
                    }
                )
                enhanced_examples.append(enhanced_example)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate LLM prompt: {e}")
                enhanced_examples.append(example)
        
        return enhanced_examples
    
    def _deduplicate_examples(self, examples: List[CodeExample]) -> List[CodeExample]:
        """Remove duplicate examples based on content hash."""
        seen_hashes = set()
        unique_examples = []
        
        for example in examples:
            # Create hash of the output code (normalized)
            normalized_code = self._normalize_code(example.output)
            code_hash = hashlib.md5(normalized_code.encode()).hexdigest()
            
            if code_hash not in seen_hashes:
                seen_hashes.add(code_hash)
                unique_examples.append(example)
        
        self.logger.info(f"Deduplicated: {len(examples)} -> {len(unique_examples)} examples")
        return unique_examples
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for deduplication."""
        # Remove extra whitespace and normalize indentation
        lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):  # Skip comments
                lines.append(stripped)
        return '\n'.join(lines)
    
    def _filter_quality_examples(self, examples: List[CodeExample]) -> List[CodeExample]:
        """Filter examples based on quality metrics."""
        quality_examples = []
        
        for example in examples:
            # Quality filters
            lines = len(example.output.split('\n'))
            
            # Skip examples that are too short or too long
            if lines < 3 or lines > 100:
                continue
            
            # Skip examples with very generic names
            name = example.metadata.get('name', '').lower()
            if name in ['test', 'temp', 'tmp', 'foo', 'bar', 'example', 'demo']:
                continue
            
            # Skip examples with minimal content
            non_empty_lines = sum(1 for line in example.output.split('\n') if line.strip())
            if non_empty_lines < 2:
                continue
            
            # Prefer examples with docstrings for better style learning
            if example.metadata.get('has_docstring'):
                example.metadata['priority'] = 'high'
            
            # Prefer examples with good naming conventions
            if example.metadata.get('naming_style') in ['snake_case', 'camelCase', 'PascalCase']:
                example.metadata['priority'] = 'high'
            
            quality_examples.append(example)
        
        # Sort by priority and complexity for better training
        quality_examples.sort(key=lambda x: (
            x.metadata.get('priority') == 'high',
            x.metadata.get('complexity', 0),
            x.metadata.get('has_docstring', False)
        ), reverse=True)
        
        self.logger.info(f"Quality filtering: {len(examples)} -> {len(quality_examples)} examples")
        return quality_examples
    
    def _convert_to_qwen_format(self, examples: List[CodeExample]) -> List[Dict[str, Any]]:
        """Convert examples to Qwen2.5 chat format."""
        dataset = []
        
        for example in examples:
            # Qwen2.5 uses ChatML format
            dataset.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Qwen2.5-Coder, a helpful assistant that writes code following specific project conventions and style guidelines."
                    },
                    {
                        "role": "user", 
                        "content": f"{example.instruction}\n\nContext: {example.input_context}" if example.input_context else example.instruction
                    },
                    {
                        "role": "assistant",
                        "content": example.output
                    }
                ],
                "metadata": example.metadata
            })
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str):
        """Save dataset in JSONL format (preferred for Qwen2.5 fine-tuning)."""
        # Save as JSONL for better memory efficiency during training
        jsonl_path = output_path.replace('.json', '.jsonl')
        
        create_output_directory(output_path)

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                # Remove metadata for training (keep it separate)
                training_item = {k: v for k, v in item.items() if k != 'metadata'}
                json.dump(training_item, f, ensure_ascii=False)
                f.write('\n')
        
        self.logger.info(f"Dataset saved to: {jsonl_path}")
        
        # Save metadata separately
        metadata_path = output_path.replace('.json', '_metadata.jsonl')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                if 'metadata' in item:
                    json.dump(item['metadata'], f, ensure_ascii=False)
                    f.write('\n')
        
        # Also save a sample for inspection
        sample_path = output_path.replace('.json', '_sample.json')
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(dataset[:10], f, indent=2, ensure_ascii=False)
        self.logger.info(f"Sample saved to: {sample_path}")
    
    def generate_summary(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive dataset summary."""
        summary = {
            "total_examples": len(dataset),
            "optimization": "QLoRA fine-tuning for Qwen2.5-Coder",
            "style_focus": True,
            "example_types": {},
            "languages": {},
            "style_patterns": self.style_analyzer.get_patterns(),
            "files_processed": set(),
            "quality_metrics": {
                "avg_lines_per_example": 0,
                "examples_with_docstrings": 0,
                "high_priority_examples": 0,
                "llm_generated_prompts": 0
            }
        }
        
        total_lines = 0
        for item in dataset:
            metadata = item.get("metadata", {})
            
            # Count example types
            example_type = metadata.get("type", "unknown")
            summary["example_types"][example_type] = summary["example_types"].get(example_type, 0) + 1
            
            # Count languages
            language = metadata.get("language", Path(metadata.get("file", "")).suffix)
            summary["languages"][language] = summary["languages"].get(language, 0) + 1
            
            # Track files
            if "file" in metadata:
                summary["files_processed"].add(metadata["file"])
            
            # Quality metrics
            if metadata.get("has_docstring"):
                summary["quality_metrics"]["examples_with_docstrings"] += 1
            if metadata.get("priority") == "high":
                summary["quality_metrics"]["high_priority_examples"] += 1
            if metadata.get("llm_generated_prompt"):
                summary["quality_metrics"]["llm_generated_prompts"] += 1
                
            # Calculate average lines
            if "messages" in item:
                assistant_msg = next((msg for msg in item["messages"] if msg["role"] == "assistant"), {})
                total_lines += len(assistant_msg.get("content", "").split('\n'))
        
        summary["quality_metrics"]["avg_lines_per_example"] = total_lines / len(dataset) if dataset else 0
        summary["files_processed"] = len(summary["files_processed"])
        
        return summary