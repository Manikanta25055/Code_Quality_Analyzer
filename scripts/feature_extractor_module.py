"""
Feature Extraction Module (Standalone)
Reusable feature extractor without main execution.

Author: Manikanta Gonugondla
Co-Author: Claude Sonnet 4.5
"""

import ast
import os
import numpy as np
import requests
import zipfile
import io


class CodeAnalyzer:
    """Analyzes Python code and extracts quality features"""

    def __init__(self, code_text):
        self.code = code_text
        self.lines = code_text.split('\n')
        self.features = {}

        try:
            self.tree = ast.parse(code_text)
            self.parse_success = True
        except SyntaxError:
            self.tree = None
            self.parse_success = False

    def extract_all_features(self):
        """Extract all features from code"""
        if not self.parse_success:
            return self._get_default_features()

        self._extract_structural_features()
        self._extract_naming_features()
        self._extract_pattern_features()
        self._extract_maintainability_features()

        return self.features

    def _extract_structural_features(self):
        """Extract structural code features"""
        functions = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]

        self.features['num_functions'] = len(functions)
        self.features['num_classes'] = len(classes)

        # Average function length
        func_lengths = []
        for func in functions:
            func_lines = func.end_lineno - func.lineno if hasattr(func, 'end_lineno') else 0
            func_lengths.append(func_lines)

        self.features['avg_function_length'] = np.mean(func_lengths) if func_lengths else 0
        self.features['max_function_length'] = max(func_lengths) if func_lengths else 0

        # Average class size
        class_sizes = []
        for cls in classes:
            class_lines = cls.end_lineno - cls.lineno if hasattr(cls, 'end_lineno') else 0
            class_sizes.append(class_lines)

        self.features['avg_class_size'] = np.mean(class_sizes) if class_sizes else 0

        # Maximum nesting depth
        self.features['max_nesting_depth'] = self._calculate_max_nesting()

        # Average parameters per function
        param_counts = [len(func.args.args) for func in functions]
        self.features['avg_params_per_function'] = np.mean(param_counts) if param_counts else 0

        # Cyclomatic complexity
        self.features['cyclomatic_complexity'] = self._calculate_complexity()

        # Lines of code
        self.features['total_lines'] = len(self.lines)
        self.features['code_lines'] = len([line for line in self.lines if line.strip() and not line.strip().startswith('#')])

    def _extract_naming_features(self):
        """Extract naming convention features"""
        all_names = []
        bad_names = 0

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                all_names.append(node.id)
                if len(node.id) == 1 and node.id.islower() and node.id not in ['i', 'j', 'k', 'x', 'y', 'z']:
                    bad_names += 1

        self.features['total_variables'] = len(all_names)
        self.features['bad_var_name_ratio'] = bad_names / len(all_names) if all_names else 0
        self.features['avg_variable_name_length'] = np.mean([len(n) for n in all_names]) if all_names else 0

        # Naming consistency
        snake_case = sum(1 for n in all_names if '_' in n)
        camel_case = sum(1 for n in all_names if n[0].islower() and any(c.isupper() for c in n))
        self.features['naming_consistency'] = max(snake_case, camel_case) / len(all_names) if all_names else 0

    def _extract_pattern_features(self):
        """Extract code pattern features"""
        # Docstrings
        docstring_count = 0
        functions = [n for n in ast.walk(self.tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]

        for node in functions:
            if ast.get_docstring(node):
                docstring_count += 1

        self.features['docstring_ratio'] = docstring_count / len(functions) if functions else 0

        # Type hints
        typed_functions = [n for n in ast.walk(self.tree)
                          if isinstance(n, ast.FunctionDef) and n.returns is not None]
        all_functions = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)]
        self.features['type_hint_ratio'] = len(typed_functions) / len(all_functions) if all_functions else 0

        # Error handling
        try_blocks = [n for n in ast.walk(self.tree) if isinstance(n, ast.Try)]
        self.features['try_except_count'] = len(try_blocks)
        self.features['error_handling_ratio'] = len(try_blocks) / len(all_functions) if all_functions else 0

        # Magic numbers
        numbers = [n for n in ast.walk(self.tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
        self.features['magic_number_count'] = len([n for n in numbers if n.value not in [0, 1, -1]])

        # Comment ratio
        comment_lines = len([line for line in self.lines if line.strip().startswith('#')])
        self.features['comment_ratio'] = comment_lines / len(self.lines) if self.lines else 0

    def _extract_maintainability_features(self):
        """Extract maintainability metrics"""
        functions = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)]
        return_counts = []

        for func in functions:
            returns = [n for n in ast.walk(func) if isinstance(n, ast.Return)]
            return_counts.append(len(returns))

        self.features['avg_returns_per_function'] = np.mean(return_counts) if return_counts else 0

        # Import count
        imports = [n for n in ast.walk(self.tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.features['import_count'] = len(imports)

    def _calculate_max_nesting(self):
        """Calculate maximum nesting depth"""
        def get_depth(node, current=0):
            if isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                depths = [get_depth(child, current + 1) for child in ast.iter_child_nodes(node)]
                return max(depths) if depths else current + 1
            else:
                depths = [get_depth(child, current) for child in ast.iter_child_nodes(node)]
                return max(depths) if depths else current

        return get_depth(self.tree)

    def _calculate_complexity(self):
        """Calculate cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler)):
                complexity += 1
        return complexity

    def _get_default_features(self):
        """Return default features for unparseable code"""
        return {
            'num_functions': 0,
            'num_classes': 0,
            'avg_function_length': 0,
            'max_function_length': 0,
            'avg_class_size': 0,
            'max_nesting_depth': 0,
            'avg_params_per_function': 0,
            'cyclomatic_complexity': 0,
            'total_lines': len(self.lines),
            'code_lines': 0,
            'total_variables': 0,
            'bad_var_name_ratio': 0,
            'avg_variable_name_length': 0,
            'naming_consistency': 0,
            'docstring_ratio': 0,
            'type_hint_ratio': 0,
            'try_except_count': 0,
            'error_handling_ratio': 0,
            'magic_number_count': 0,
            'comment_ratio': 0,
            'avg_returns_per_function': 0,
            'import_count': 0
        }


class RepoFeatureExtractor:
    """Extracts features from entire repositories"""

    def __init__(self, output_dir='../data/raw_repos'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_repo(self, repo_name, clone_url):
        """Download repository as ZIP from GitHub"""
        try:
            # Extract owner and repo from URL
            # clone_url format: https://github.com/owner/repo.git
            if 'github.com' in clone_url:
                parts = clone_url.replace('.git', '').split('/')
                owner = parts[-2]
                repo = parts[-1]

                # Method 1: Try default branch using GitHub API
                print(f"  Getting default branch info...")
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                try:
                    api_response = requests.get(api_url, timeout=10)
                    if api_response.status_code == 200:
                        repo_info = api_response.json()
                        default_branch = repo_info.get('default_branch', 'main')
                        print(f"  Default branch: {default_branch}")

                        # Download using default branch
                        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{default_branch}.zip"
                        response = requests.get(zip_url, timeout=30)

                        if response.status_code == 200:
                            print(f"  Successfully downloaded from {default_branch} branch")
                            return io.BytesIO(response.content)
                except Exception as e:
                    print(f"  API method failed: {e}")

                # Method 2: Try common branch names
                branches = ['main', 'master', 'develop']
                for branch in branches:
                    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
                    print(f"  Trying branch: {branch}")

                    try:
                        response = requests.get(zip_url, timeout=30)

                        if response.status_code == 200:
                            print(f"  Successfully downloaded from {branch} branch")
                            return io.BytesIO(response.content)
                        elif response.status_code == 404:
                            print(f"  Branch {branch} not found")
                            continue

                    except Exception as e:
                        print(f"  Error on {branch}: {e}")
                        continue

                # Method 3: Try direct zipball download
                print(f"  Trying direct zipball download...")
                zipball_url = f"https://github.com/{owner}/{repo}/zipball"
                try:
                    response = requests.get(zipball_url, timeout=30, allow_redirects=True)
                    if response.status_code == 200:
                        print(f"  Successfully downloaded via zipball")
                        return io.BytesIO(response.content)
                except Exception as e:
                    print(f"  Zipball method failed: {e}")

            print(f"  Failed to download from any method")
            return None

        except Exception as e:
            print(f"Error downloading {repo_name}: {e}")
            return None

    def extract_python_files(self, zip_content, max_files=20):
        """Extract Python files from ZIP"""
        python_files = []

        try:
            with zipfile.ZipFile(zip_content) as zf:
                for file_info in zf.filelist[:100]:
                    if file_info.filename.endswith('.py') and not file_info.is_dir():
                        try:
                            content = zf.read(file_info.filename).decode('utf-8', errors='ignore')
                            python_files.append(content)

                            if len(python_files) >= max_files:
                                break
                        except:
                            continue

        except Exception as e:
            print(f"Error extracting files: {e}")

        return python_files

    def extract_repo_features(self, repo_name, clone_url):
        """Extract features from entire repository"""
        print(f"Processing: {repo_name}")

        zip_content = self.download_repo(repo_name, clone_url)
        if not zip_content:
            print(f"  Failed to download")
            return None

        python_files = self.extract_python_files(zip_content, max_files=20)
        if not python_files:
            print(f"  No Python files found")
            return None

        print(f"  Analyzing {len(python_files)} Python files")

        all_features = []
        for file_content in python_files:
            analyzer = CodeAnalyzer(file_content)
            features = analyzer.extract_all_features()
            all_features.append(features)

        # Average features across files
        aggregated = {}
        feature_keys = all_features[0].keys()

        for key in feature_keys:
            values = [f[key] for f in all_features if key in f]
            aggregated[key] = np.mean(values) if values else 0

        return aggregated
