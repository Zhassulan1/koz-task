import ast
import re
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .style_analyzer import StyleAnalyzer


class BaseParser(ABC):
    """Base class for language-specific parsers."""

    def __init__(self, style_analyzer: StyleAnalyzer):
        self.style_analyzer = style_analyzer
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def parse_file(self, file_path: Path) -> List['CodeExample']:
        """Parse a file and return code examples."""
        pass

    def _calculate_complexity(self, source: str) -> int:
        """Calculate a simple complexity score for code."""
        complexity = 0
        keywords = ['if', 'for', 'while', 'try', 'except', 'with', 'def', 'class', 'function', 'switch', 'case']
        source_lower = source.lower()
        for keyword in keywords:
            complexity += source_lower.count(keyword)
        return complexity

    def _detect_naming_style(self, name: str) -> str:
        """Detect the naming convention used."""
        if not name or not name.replace('_', '').isalnum():
            return 'mixed'

        if '_' in name and name.islower():
            return 'snake_case'
        elif name[0].isupper() and '_' not in name:
            return 'PascalCase'
        elif name[0].islower() and any(c.isupper() for c in name[1:]):
            return 'camelCase'
        elif name.isupper() and '_' in name:
            return 'UPPER_CASE'
        elif name.islower():
            return 'lowercase'
        else:
            return 'mixed'


class PythonParser(BaseParser):
    """Parser for Python files using AST."""
    
    def parse_file(self, file_path: Path) -> List['CodeExample']:
        """Parse Python file and extract training examples."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    examples.extend(self._extract_function_examples(node, file_path))
                elif isinstance(node, ast.ClassDef):
                    examples.extend(self._extract_class_examples(node, file_path))
                    
        except Exception as e:
            self.logger.warning(f"Error parsing Python file {file_path}: {e}")
            
        return examples
    
    def _extract_function_examples(self, node: ast.FunctionDef, file_path: Path) -> List['CodeExample']:
        """Extract function-related training examples."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        try:
            source = ast.unparse(node)
            func_name = node.name
            docstring = ast.get_docstring(node)
            
            # Skip special methods and very short functions
            if func_name.startswith('__') or len(source.split('\n')) < 3:
                return examples
            
            # Get function arguments
            args = []
            for arg in node.args.args:
                arg_name = arg.arg
                type_annotation = ""
                if arg.annotation:
                    type_annotation = ast.unparse(arg.annotation)
                args.append(f"{arg_name}: {type_annotation}" if type_annotation else arg_name)
            
            # Create style context
            style_context = self.style_analyzer.get_style_context(file_path)
            
            # Calculate metrics
            lines = len(source.split('\n'))
            complexity_score = self._calculate_complexity(source)
            
            # Skip overly complex functions
            if complexity_score > 20 or lines > 50:
                return examples
            
            # Generate instruction based on function analysis
            instruction = self._generate_function_instruction(func_name, docstring, args)
            
            # Main function example
            examples.append(CodeExample(
                instruction=instruction,
                input_context=f"File: {file_path.name} | {style_context}",
                output=source,
                metadata={
                    'type': 'python_function',
                    'name': func_name,
                    'file': str(file_path.relative_to(file_path.parents[1])),
                    'language': 'python',
                    'lines': lines,
                    'complexity': complexity_score,
                    'has_docstring': bool(docstring),
                    'arg_count': len(args),
                    'naming_style': self._detect_naming_style(func_name),
                    'style_focus': True
                }
            ))
            
            # Additional examples for well-documented functions
            if docstring and len(docstring) > 20:
                examples.append(CodeExample(
                    instruction="Write a well-documented function following this project's documentation style",
                    input_context=f"Purpose: {docstring.split('.')[0]}",
                    output=source,
                    metadata={
                        'type': 'python_documented_function',
                        'name': func_name,
                        'file': str(file_path.relative_to(file_path.parents[1])),
                        'language': 'python',
                        'doc_style': self._analyze_docstring_style(docstring),
                        'style_focus': True
                    }
                ))
            
        except Exception as e:
            self.logger.debug(f"Error extracting function {node.name}: {e}")
        
        return examples
    
    def _extract_class_examples(self, node: ast.ClassDef, file_path: Path) -> List['CodeExample']:
        """Extract class-related training examples."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        try:
            source = ast.unparse(node)
            class_name = node.name
            docstring = ast.get_docstring(node)
            
            # Skip very large classes
            if len(source.split('\n')) > 100:
                return examples
            
            # Get methods
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            method_names = [m.name for m in methods if not m.name.startswith('__')]
            
            # Create style context
            style_context = self.style_analyzer.get_style_context(file_path)
            
            # Generate class instruction
            instruction = self._generate_class_instruction(class_name, docstring, method_names)
            
            # Main class example
            examples.append(CodeExample(
                instruction=instruction,
                input_context=f"File: {file_path.name} | {style_context}",
                output=source,
                metadata={
                    'type': 'python_class',
                    'name': class_name,
                    'file': str(file_path.relative_to(file_path.parents[1])),
                    'language': 'python',
                    'method_count': len(methods),
                    'has_docstring': bool(docstring),
                    'naming_style': self._detect_naming_style(class_name),
                    'style_focus': True
                }
            ))
            
        except Exception as e:
            self.logger.debug(f"Error extracting class {node.name}: {e}")
        
        return examples
    
    def _generate_function_instruction(self, name: str, docstring: Optional[str], args: List[str]) -> str:
        """Generate appropriate instruction for function."""
        if docstring:
            purpose = docstring.split('.')[0].lower()
            if purpose:
                return f"Write a function that {purpose}"
        
        # Fallback based on name analysis
        if 'get' in name.lower():
            return f"Create a getter function named {name}"
        elif 'set' in name.lower():
            return f"Create a setter function named {name}"
        elif 'calculate' in name.lower() or 'compute' in name.lower():
            return f"Write a calculation function named {name}"
        elif 'validate' in name.lower() or 'check' in name.lower():
            return f"Create a validation function named {name}"
        else:
            return f"Implement a function named {name}"
    
    def _generate_class_instruction(self, name: str, docstring: Optional[str], methods: List[str]) -> str:
        """Generate appropriate instruction for class."""
        if docstring:
            purpose = docstring.split('.')[0].lower()
            if purpose:
                return f"Create a class that {purpose}"
        
        # Fallback based on name and methods analysis
        if 'manager' in name.lower():
            return f"Design a {name} class for managing resources"
        elif 'handler' in name.lower():
            return f"Create a {name} class for handling operations"
        elif methods and 'process' in ' '.join(methods).lower():
            return f"Implement a {name} class with processing capabilities"
        else:
            return f"Create a {name} class with appropriate methods"
    
    def _analyze_docstring_style(self, docstring: str) -> str:
        """Analyze docstring style."""
        if 'Args:' in docstring and 'Returns:' in docstring:
            return 'google_style'
        elif ':param' in docstring and ':return' in docstring:
            return 'sphinx_style'
        else:
            return 'basic'


class JavaScriptParser(BaseParser):
    """Parser for JavaScript/TypeScript files using regex."""
    
    def parse_file(self, file_path: Path) -> List['CodeExample']:
        """Parse JavaScript/TypeScript file and extract examples."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract different types of functions and classes
            examples.extend(self._extract_functions(content, file_path))
            examples.extend(self._extract_arrow_functions(content, file_path))
            examples.extend(self._extract_classes(content, file_path))
            
        except Exception as e:
            self.logger.warning(f"Error parsing JavaScript file {file_path}: {e}")
        
        return examples
    
    def _extract_functions(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract regular function declarations."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'function\s+(\w+)\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            func_name = match.group(1)
            func_source = match.group(0)
            
            if len(func_source.split('\n')) >= 3:
                examples.append(CodeExample(
                    instruction=f"Write a JavaScript function named {func_name}",
                    input_context=f"File: {file_path.name} | {self.style_analyzer.get_style_context(file_path)}",
                    output=func_source,
                    metadata={
                        'type': 'javascript_function',
                        'name': func_name,
                        'file': str(file_path.relative_to(file_path.parents[1])),
                        'language': 'javascript',
                        'naming_style': self._detect_naming_style(func_name),
                        'style_focus': True
                    }
                ))
        
        return examples
    
    def _extract_arrow_functions(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract arrow function declarations."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            func_name = match.group(1)
            func_source = match.group(0)
            
            if len(func_source.split('\n')) >= 3:
                examples.append(CodeExample(
                    instruction=f"Create an arrow function named {func_name}",
                    input_context=f"File: {file_path.name} | Modern JavaScript style",
                    output=func_source,
                    metadata={
                        'type': 'javascript_arrow_function',
                        'name': func_name,
                        'file': str(file_path.relative_to(file_path.parents[1])),
                        'language': 'javascript',
                        'naming_style': self._detect_naming_style(func_name),
                        'style_focus': True
                    }
                ))
        
        return examples
    
    def _extract_classes(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract class declarations."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            class_name = match.group(1)
            class_source = match.group(0)
            
            if len(class_source.split('\n')) >= 5:
                examples.append(CodeExample(
                    instruction=f"Create a JavaScript class named {class_name}",
                    input_context=f"File: {file_path.name} | {self.style_analyzer.get_style_context(file_path)}",
                    output=class_source,
                    metadata={
                        'type': 'javascript_class',
                        'name': class_name,
                        'file': str(file_path.relative_to(file_path.parents[1])),
                        'language': 'javascript',
                        'naming_style': self._detect_naming_style(class_name),
                        'style_focus': True
                    }
                ))
        
        return examples


class JavaParser(BaseParser):
    """Parser for Java files using regex."""
    
    def parse_file(self, file_path: Path) -> List['CodeExample']:
        """Parse Java file and extract examples."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            examples.extend(self._extract_methods(content, file_path))
            examples.extend(self._extract_classes(content, file_path))
            
        except Exception as e:
            self.logger.warning(f"Error parsing Java file {file_path}: {e}")
        
        return examples
    
    def _extract_methods(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract Java method declarations."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'(public|private|protected)\s+(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:\s*,\s*\w+)*)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            method_name = match.group(2)
            method_source = match.group(0)
            
            # Skip constructors and very short methods
            if method_name[0].isupper() or len(method_source.split('\n')) < 4:
                continue
            
            examples.append(CodeExample(
                instruction=f"Write a Java method named {method_name}",
                input_context=f"File: {file_path.name} | Java method",
                output=method_source,
                metadata={
                    'type': 'java_method',
                    'name': method_name,
                    'file': str(file_path.relative_to(file_path.parents[1])),
                    'language': 'java',
                    'naming_style': self._detect_naming_style(method_name),
                    'style_focus': True
                }
            ))
        
        return examples
    
    def _extract_classes(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract Java class declarations."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'(?:public\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+(?:\s*,\s*\w+)*)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            class_name = match.group(1)
            class_source = match.group(0)
            
            if len(class_source.split('\n')) >= 8:
                examples.append(CodeExample(
                    instruction=f"Create a Java class named {class_name}",
                    input_context=f"File: {file_path.name} | Java class structure",
                    output=class_source,
                    metadata={
                        'type': 'java_class',
                        'name': class_name,
                        'file': str(file_path.relative_to(file_path.parents[1])),
                        'language': 'java',
                        'naming_style': self._detect_naming_style(class_name),
                        'style_focus': True
                    }
                ))
        
        return examples


class CppParser(BaseParser):
    """Parser for C++ files using regex."""
    
    def parse_file(self, file_path: Path) -> List['CodeExample']:
        """Parse C++ file and extract examples."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            examples.extend(self._extract_functions(content, file_path))
            examples.extend(self._extract_classes(content, file_path))
            
        except Exception as e:
            self.logger.warning(f"Error parsing C++ file {file_path}: {e}")
        
        return examples
    
    def _extract_functions(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract C++ function definitions."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'\b(?:inline\s+)?\w+(?:\s*\*+|\s*&+)?\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            func_name = match.group(1)
            func_source = match.group(0)
            
            # Skip common keywords and very short functions
            if (func_name.lower() in ['if', 'for', 'while', 'switch', 'return'] or 
                len(func_source.split('\n')) < 4):
                continue
            
            examples.append(CodeExample(
                instruction=f"Write a C++ function named {func_name}",
                input_context=f"File: {file_path.name} | C++ implementation",
                output=func_source,
                metadata={
                    'type': 'cpp_function',
                    'name': func_name,
                    'file': str(file_path.relative_to(file_path.parents[1])),
                    'language': 'cpp',
                    'naming_style': self._detect_naming_style(func_name),
                    'style_focus': True
                }
            ))
        
        return examples
    
    def _extract_classes(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract C++ class definitions."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            class_name = match.group(1)
            class_source = match.group(0)
            
            if len(class_source.split('\n')) >= 6:
                examples.append(CodeExample(
                    instruction=f"Create a C++ class named {class_name}",
                    input_context=f"File: {file_path.name} | C++ class definition",
                    output=class_source,
                    metadata={
                        'type': 'cpp_class',
                        'name': class_name,
                        'file': str(file_path.relative_to(file_path.parents[1])),
                        'language': 'cpp',
                        'naming_style': self._detect_naming_style(class_name),
                        'style_focus': True
                    }
                ))
        
        return examples


class GoParser(BaseParser):
    """Parser for Go files using regex."""
    
    def parse_file(self, file_path: Path) -> List['CodeExample']:
        """Parse Go file and extract examples."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            examples.extend(self._extract_functions(content, file_path))
            examples.extend(self._extract_structs(content, file_path))
            
        except Exception as e:
            self.logger.warning(f"Error parsing Go file {file_path}: {e}")
        
        return examples
    
    def _extract_functions(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract Go function definitions."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)(?:\s*\([^)]*\)|\s*\w+)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            func_name = match.group(1)
            func_source = match.group(0)
            
            if len(func_source.split('\n')) >= 3:
                examples.append(CodeExample(
                    instruction=f"Write a Go function named {func_name}",
                    input_context=f"File: {file_path.name} | Go function",
                    output=func_source,
                    metadata={
                        'type': 'go_function',
                        'name': func_name,
                        'file': str(file_path.relative_to(file_path.parents[1])),
                        'language': 'go',
                        'naming_style': self._detect_naming_style(func_name),
                        'style_focus': True
                    }
                ))
        
        return examples
    
    def _extract_structs(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract Go struct definitions."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'type\s+(\w+)\s+struct\s*\{[^{}]*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            struct_name = match.group(1)
            struct_source = match.group(0)
            
            examples.append(CodeExample(
                instruction=f"Define a Go struct named {struct_name}",
                input_context=f"File: {file_path.name} | Go struct definition",
                output=struct_source,
                metadata={
                    'type': 'go_struct',
                    'name': struct_name,
                    'file': str(file_path.relative_to(file_path.parents[1])),
                    'language': 'go',
                    'naming_style': self._detect_naming_style(struct_name),
                    'style_focus': True
                }
            ))
        
        return examples


class RustParser(BaseParser):
    """Parser for Rust files using regex."""
    
    def parse_file(self, file_path: Path) -> List['CodeExample']:
        """Parse Rust file and extract examples."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            examples.extend(self._extract_functions(content, file_path))
            examples.extend(self._extract_structs(content, file_path))
            
        except Exception as e:
            self.logger.warning(f"Error parsing Rust file {file_path}: {e}")
        
        return examples
    
    def _extract_functions(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract Rust function definitions."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'(?:pub\s+)?fn\s+(\w+)(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            func_name = match.group(1)
            func_source = match.group(0)
            
            if len(func_source.split('\n')) >= 3:
                examples.append(CodeExample(
                    instruction=f"Write a Rust function named {func_name}",
                    input_context=f"File: {file_path.name} | Rust function",
                    output=func_source,
                    metadata={
                        'type': 'rust_function',
                        'name': func_name,
                        'file': str(file_path.relative_to(file_path.parents[1])),
                        'language': 'rust',
                        'naming_style': self._detect_naming_style(func_name),
                        'style_focus': True
                    }
                ))
        
        return examples
    
    def _extract_structs(self, content: str, file_path: Path) -> List['CodeExample']:
        """Extract Rust struct definitions."""
        from .qwen_dataset_generator import CodeExample
        examples = []
        
        pattern = r'(?:pub\s+)?struct\s+(\w+)(?:<[^>]*>)?\s*\{[^{}]*\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            struct_name = match.group(1)
            struct_source = match.group(0)

            examples.append(CodeExample(
                instruction=f"Define a Rust struct named {struct_name}",
                input_context=f"File: {file_path.name} | Rust struct definition",
                output=struct_source,
                metadata={
                    'type': 'rust_struct',
                    'name': struct_name,
                    'file': str(file_path.relative_to(file_path.parents[1])),
                    'language': 'rust',
                    'naming_style': self._detect_naming_style(struct_name),
                    'style_focus': True
                }
            ))

        return examples
