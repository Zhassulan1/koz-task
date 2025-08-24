import ast
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


class StyleAnalyzer:
    """Analyzes code repositories to extract consistent style patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = {
            'naming_conventions': {
                'functions': Counter(),
                'classes': Counter(),
                'variables': Counter(),
                'constants': Counter()
            },
            'import_patterns': [],
            'docstring_styles': Counter(),
            'indentation': Counter(),
            'line_length': [],
            'error_handling_patterns': [],
            'comment_styles': Counter(),
            'file_organization': {
                'imports_first': 0,
                'classes_after_functions': 0,
                'main_at_end': 0
            }
        }
        self.analyzed_files = 0
    
    def analyze_repository(self, files: List[Path]) -> None:
        """Analyze multiple files to extract style patterns."""
        self.logger.info(f"Analyzing style patterns from {len(files)} files")
        
        for file_path in files:
            try:
                self._analyze_file(file_path)
                self.analyzed_files += 1
            except Exception as e:
                self.logger.debug(f"Error analyzing {file_path}: {e}")
        
        self._consolidate_patterns()
        self.logger.info(f"Style analysis complete. Analyzed {self.analyzed_files} files")
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for style patterns."""
        ext = file_path.suffix.lower()
        
        if ext == '.py':
            self._analyze_python_file(file_path)
        elif ext in ['.js', '.ts']:
            self._analyze_javascript_file(file_path)
        elif ext == '.java':
            self._analyze_java_file(file_path)
        # Add more language-specific analyzers as needed
    
    def _analyze_python_file(self, file_path: Path) -> None:
        """Analyze Python-specific style patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            tree = ast.parse(content)
            
            # Analyze naming conventions
            self._analyze_python_naming(tree)
            
            # Analyze imports
            self._analyze_python_imports(tree, lines)
            
            # Analyze docstrings
            self._analyze_python_docstrings(tree)
            
            # Analyze indentation
            self._analyze_indentation(lines)
            
            # Analyze line lengths
            self._analyze_line_lengths(lines)
            
            # Analyze file organization
            self._analyze_python_file_organization(tree, lines)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing Python file {file_path}: {e}")
    
    def _analyze_python_naming(self, tree: ast.AST) -> None:
        """Analyze Python naming conventions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                style = self._detect_naming_style(node.name)
                self.patterns['naming_conventions']['functions'][style] += 1
                
            elif isinstance(node, ast.ClassDef):
                style = self._detect_naming_style(node.name)
                self.patterns['naming_conventions']['classes'][style] += 1
                
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():
                            self.patterns['naming_conventions']['constants']['UPPER_CASE'] += 1
                        else:
                            style = self._detect_naming_style(target.id)
                            self.patterns['naming_conventions']['variables'][style] += 1
    
    def _analyze_python_imports(self, tree: ast.AST, lines: List[str]) -> None:
        """Analyze import patterns."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = [alias.name for alias in node.names]
                imports.append(f"from {module} import {', '.join(names)}")
        
        # Store first few import patterns as examples
        if imports and len(self.patterns['import_patterns']) < 50:
            self.patterns['import_patterns'].extend(imports[:5])
    
    def _analyze_python_docstrings(self, tree: ast.AST) -> None:
        """Analyze docstring styles."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    style = self._detect_docstring_style(docstring)
                    self.patterns['docstring_styles'][style] += 1
    
    def _analyze_indentation(self, lines: List[str]) -> None:
        """Analyze indentation patterns."""
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_space = len(line) - len(line.lstrip())
                if leading_space > 0:
                    # Detect if using tabs or spaces
                    if line.startswith('\t'):
                        self.patterns['indentation']['tabs'] += 1
                    else:
                        self.patterns['indentation']['spaces'] += 1
                        # Count space increments (2, 4, 8, etc.)
                        self.patterns['indentation'][f'{leading_space}_spaces'] += 1
    
    def _analyze_line_lengths(self, lines: List[str]) -> None:
        """Analyze line length patterns."""
        for line in lines:
            length = len(line.rstrip())
            if length > 0:
                self.patterns['line_length'].append(length)
    
    def _analyze_python_file_organization(self, tree: ast.AST, lines: List[str]) -> None:
        """Analyze file organization patterns."""
        # Check if imports come first
        first_statement = None
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                first_statement = node
                break
        
        if first_statement is None or isinstance(tree.body[0], (ast.Import, ast.ImportFrom)):
            self.patterns['file_organization']['imports_first'] += 1
        
        # Check for main block at end
        last_lines = '\n'.join(lines[-10:])
        if 'if __name__ == "__main__"' in last_lines:
            self.patterns['file_organization']['main_at_end'] += 1
    
    def _analyze_javascript_file(self, file_path: Path) -> None:
        """Analyze JavaScript/TypeScript style patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Analyze function naming
            func_patterns = [
                r'function\s+(\w+)',
                r'const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
                r'(\w+)\s*:\s*(?:async\s+)?function'
            ]
            
            for pattern in func_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    func_name = match.group(1)
                    style = self._detect_naming_style(func_name)
                    self.patterns['naming_conventions']['functions'][style] += 1
            
            # Analyze class naming
            class_matches = re.finditer(r'class\s+(\w+)', content)
            for match in class_matches:
                class_name = match.group(1)
                style = self._detect_naming_style(class_name)
                self.patterns['naming_conventions']['classes'][style] += 1
            
            # Analyze indentation and line lengths
            self._analyze_indentation(lines)
            self._analyze_line_lengths(lines)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing JavaScript file {file_path}: {e}")
    
    def _analyze_java_file(self, file_path: Path) -> None:
        """Analyze Java style patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Analyze method naming
            method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?\w+\s+(\w+)\s*\('
            matches = re.finditer(method_pattern, content)
            for match in matches:
                method_name = match.group(1)
                style = self._detect_naming_style(method_name)
                self.patterns['naming_conventions']['functions'][style] += 1
            
            # Analyze class naming
            class_matches = re.finditer(r'(?:public\s+)?class\s+(\w+)', content)
            for match in class_matches:
                class_name = match.group(1)
                style = self._detect_naming_style(class_name)
                self.patterns['naming_conventions']['classes'][style] += 1
            
            self._analyze_indentation(lines)
            self._analyze_line_lengths(lines)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing Java file {file_path}: {e}")
    
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
    
    def _detect_docstring_style(self, docstring: str) -> str:
        """Analyze docstring style patterns."""
        if not docstring:
            return 'none'
        
        # Check for common docstring formats
        if 'Args:' in docstring and 'Returns:' in docstring:
            return 'google_style'
        elif ':param' in docstring and ':return' in docstring:
            return 'sphinx_style'
        elif docstring.count('"""') >= 2 or docstring.count("'''") >= 2:
            return 'triple_quote'
        elif len(docstring.split('\n')) == 1:
            return 'single_line'
        else:
            return 'multi_line'
    
    def _consolidate_patterns(self) -> None:
        """Consolidate and clean up detected patterns."""
        # Find most common patterns
        for category in ['functions', 'classes', 'variables']:
            if self.patterns['naming_conventions'][category]:
                most_common = self.patterns['naming_conventions'][category].most_common(1)[0][0]
                self.patterns['naming_conventions'][f'{category}_preferred'] = most_common
        
        # Calculate average line length
        if self.patterns['line_length']:
            avg_length = sum(self.patterns['line_length']) / len(self.patterns['line_length'])
            self.patterns['avg_line_length'] = round(avg_length, 1)
            
            # Determine preferred line length threshold
            lines_over_80 = sum(1 for length in self.patterns['line_length'] if length > 80)
            lines_over_100 = sum(1 for length in self.patterns['line_length'] if length > 100)
            
            if lines_over_100 / len(self.patterns['line_length']) > 0.1:
                self.patterns['preferred_line_length'] = 120
            elif lines_over_80 / len(self.patterns['line_length']) > 0.2:
                self.patterns['preferred_line_length'] = 100
            else:
                self.patterns['preferred_line_length'] = 80
        
        # Find preferred indentation
        if self.patterns['indentation']:
            most_common_indent = self.patterns['indentation'].most_common(1)[0][0]
            self.patterns['preferred_indentation'] = most_common_indent
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get consolidated style patterns."""
        return self.patterns
    
    def get_style_context(self, file_path: Optional[Path] = None) -> str:
        """Generate style context string for prompts."""
        context_parts = []
        
        # Naming conventions
        naming = self.patterns['naming_conventions']
        if 'functions_preferred' in naming:
            context_parts.append(f"Functions: {naming['functions_preferred']}")
        if 'classes_preferred' in naming:
            context_parts.append(f"Classes: {naming['classes_preferred']}")
        
        # Indentation
        if 'preferred_indentation' in self.patterns:
            context_parts.append(f"Indentation: {self.patterns['preferred_indentation']}")

        # Line length
        if 'preferred_line_length' in self.patterns:
            context_parts.append(f"Max line length: {self.patterns['preferred_line_length']}")

        # Docstring style
        if self.patterns['docstring_styles']:
            common_doc_style = self.patterns['docstring_styles'].most_common(1)[0][0]
            context_parts.append(f"Documentation: {common_doc_style}")

        return " | ".join(context_parts) if context_parts else "Standard conventions"

    def should_prioritize_example(self, metadata: Dict[str, Any]) -> bool:
        """Determine if an example should be prioritized based on style patterns."""
        # Prioritize examples that match detected patterns
        name = metadata.get('name', '')
        naming_style = self._detect_naming_style(name)

        example_type = metadata.get('type', '')
        if 'function' in example_type:
            preferred = self.patterns['naming_conventions'].get('functions_preferred')
            return naming_style == preferred
        elif 'class' in example_type:
            preferred = self.patterns['naming_conventions'].get('classes_preferred')
            return naming_style == preferred

        return False
