"""
Phase 8: Tkinter GUI Application
Cross-platform GUI for code quality prediction (fallback from PyQt5).

Author: Manikanta Gonugondla
Co-Author: Claude Sonnet 4.5
"""

import sys
import os
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import shutil

# Import feature extractor module
from feature_extractor_module import CodeAnalyzer


class CodeQualityGUI:
    """Main GUI window using Tkinter"""

    def __init__(self, root):
        self.root = root
        self.root.title("Code Quality Predictor - ML Coursework Project")
        self.root.geometry("900x700")

        # Model paths
        self.model_path = '../models/advanced/best_model.pkl'
        self.scaler_path = '../models/scaler.pkl'
        self.metadata_path = '../models/advanced/best_model_metadata.pkl'

        # Local clone directory
        self.clone_dir = Path('../data/temp_clones')
        self.clone_dir.mkdir(parents=True, exist_ok=True)

        self.current_result = None
        self.is_analyzing = False

        self.setup_ui()

    def setup_ui(self):
        """Setup user interface"""
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="Code Quality Predictor",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)

        subtitle_label = tk.Label(
            title_frame,
            text="ML-based GitHub Repository Quality Analysis",
            font=('Arial', 11),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()

        # Main content frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input section
        input_frame = tk.LabelFrame(main_frame, text="Repository Input", font=('Arial', 12, 'bold'), padx=10, pady=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        url_label = tk.Label(input_frame, text="GitHub URL:", font=('Arial', 10))
        url_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.url_entry = tk.Entry(input_frame, font=('Arial', 10), width=60)
        self.url_entry.grid(row=0, column=1, padx=10, pady=5)
        self.url_entry.insert(0, "https://github.com/username/repository")

        # Help button
        help_btn = tk.Button(
            input_frame,
            text="?",
            font=('Arial', 10, 'bold'),
            bg='#95a5a6',
            fg='#000000',
            padx=8,
            pady=2,
            command=self.show_help,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2
        )
        help_btn.grid(row=0, column=2, padx=5)

        # Buttons frame
        button_frame = tk.Frame(input_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.analyze_btn = tk.Button(
            button_frame,
            text="Analyze Repository",
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='#000000',
            activebackground='#229954',
            activeforeground='#000000',
            padx=20,
            pady=10,
            command=self.analyze_repository,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(
            button_frame,
            text="Save Result",
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='#000000',
            activebackground='#2980b9',
            activeforeground='#000000',
            padx=20,
            pady=10,
            command=self.save_result,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='#000000',
            activebackground='#c0392b',
            activeforeground='#000000',
            padx=20,
            pady=10,
            command=self.clear_all,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = tk.Button(
            button_frame,
            text="Export Features",
            font=('Arial', 11, 'bold'),
            bg='#9b59b6',
            fg='#000000',
            activebackground='#8e44ad',
            activeforeground='#000000',
            padx=20,
            pady=10,
            command=self.export_features,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2,
            state=tk.DISABLED
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)

        # Result frame
        result_frame = tk.LabelFrame(main_frame, text="Prediction Result", font=('Arial', 12, 'bold'), padx=10, pady=10)
        result_frame.pack(fill=tk.X, pady=(0, 10))

        self.result_label = tk.Label(
            result_frame,
            text="No analysis yet",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=20,
            pady=15
        )
        self.result_label.pack(fill=tk.X)

        # Output section - side by side
        output_frame = tk.Frame(main_frame)
        output_frame.pack(fill=tk.BOTH, expand=True)

        # Terminal output (left side)
        terminal_frame = tk.LabelFrame(output_frame, text="Terminal Output", font=('Arial', 12, 'bold'), padx=10, pady=10)
        terminal_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.terminal_output = scrolledtext.ScrolledText(
            terminal_frame,
            font=('Courier', 10),
            bg='#1e1e1e',
            fg='#00ff00',
            wrap=tk.WORD
        )
        self.terminal_output.pack(fill=tk.BOTH, expand=True)

        # Quality Issues section (right side)
        issues_frame = tk.LabelFrame(output_frame, text="Quality Issues & Recommendations", font=('Arial', 12, 'bold'), padx=10, pady=10)
        issues_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.issues_output = scrolledtext.ScrolledText(
            issues_frame,
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#ecf0f1',
            wrap=tk.WORD
        )
        self.issues_output.pack(fill=tk.BOTH, expand=True)
        self.issues_output.insert(tk.END, "Quality analysis will appear here after repository analysis...")

    def append_output(self, text):
        """Append text to terminal output"""
        self.terminal_output.insert(tk.END, text + '\n')
        self.terminal_output.see(tk.END)
        self.root.update_idletasks()

    def append_issues(self, text, color=None):
        """Append text to issues output"""
        if color:
            self.issues_output.tag_config(color, foreground=color)
            self.issues_output.insert(tk.END, text + '\n', color)
        else:
            self.issues_output.insert(tk.END, text + '\n')
        self.issues_output.see(tk.END)
        self.root.update_idletasks()

    def analyze_quality_issues(self, features, quality_score):
        """Analyze features and generate quality issues report"""
        self.issues_output.delete(1.0, tk.END)

        self.append_issues("=" * 60)
        self.append_issues("QUALITY ANALYSIS REPORT", '#3498db')
        self.append_issues("=" * 60 + "\n")

        issues = []
        recommendations = []

        # Check docstring ratio
        if features.get('docstring_ratio', 0) < 0.3:
            issues.append("LOW Documentation: Only {:.1f}% functions have docstrings".format(features.get('docstring_ratio', 0) * 100))
            recommendations.append("Add docstrings to all functions and classes")

        # Check type hints
        if features.get('type_hint_ratio', 0) < 0.2:
            issues.append("MISSING Type Hints: Only {:.1f}% functions have type hints".format(features.get('type_hint_ratio', 0) * 100))
            recommendations.append("Add type hints to function parameters and returns")

        # Check error handling
        if features.get('error_handling_ratio', 0) < 0.1:
            issues.append("POOR Error Handling: Only {:.1f}% functions have try-except blocks".format(features.get('error_handling_ratio', 0) * 100))
            recommendations.append("Implement proper error handling with try-except blocks")

        # Check function length
        if features.get('avg_function_length', 0) > 50:
            issues.append("LONG Functions: Average function length is {:.0f} lines".format(features.get('avg_function_length', 0)))
            recommendations.append("Refactor long functions into smaller, focused functions (max 30 lines)")

        # Check cyclomatic complexity
        if features.get('cyclomatic_complexity', 0) > 15:
            issues.append("HIGH Complexity: Cyclomatic complexity is {:.0f}".format(features.get('cyclomatic_complexity', 0)))
            recommendations.append("Reduce complexity by breaking down complex logic")

        # Check naming conventions
        if features.get('bad_var_name_ratio', 0) > 0.2:
            issues.append("BAD Variable Names: {:.1f}% variables have poor names".format(features.get('bad_var_name_ratio', 0) * 100))
            recommendations.append("Use descriptive variable names (avoid single letters)")

        # Check magic numbers
        if features.get('magic_number_count', 0) > 10:
            issues.append("MAGIC Numbers: Found {:.0f} hardcoded values".format(features.get('magic_number_count', 0)))
            recommendations.append("Replace magic numbers with named constants")

        # Check comment ratio
        if features.get('comment_ratio', 0) < 0.05:
            issues.append("LOW Comments: Only {:.1f}% lines have comments".format(features.get('comment_ratio', 0) * 100))
            recommendations.append("Add explanatory comments for complex logic")

        # Check nesting depth
        if features.get('max_nesting_depth', 0) > 5:
            issues.append("DEEP Nesting: Maximum nesting depth is {:.0f}".format(features.get('max_nesting_depth', 0)))
            recommendations.append("Reduce nesting depth (max 4 levels recommended)")

        # Display results
        if quality_score >= 70:
            self.append_issues("OVERALL ASSESSMENT: GOOD QUALITY", '#27ae60')
            self.append_issues("Score: {:.1f}/100\n".format(quality_score), '#27ae60')
        elif quality_score >= 50:
            self.append_issues("OVERALL ASSESSMENT: MODERATE QUALITY", '#f39c12')
            self.append_issues("Score: {:.1f}/100\n".format(quality_score), '#f39c12')
        else:
            self.append_issues("OVERALL ASSESSMENT: LOW QUALITY", '#e74c3c')
            self.append_issues("Score: {:.1f}/100\n".format(quality_score), '#e74c3c')

        if issues:
            self.append_issues("IDENTIFIED ISSUES ({}):\n".format(len(issues)), '#e74c3c')
            for i, issue in enumerate(issues, 1):
                self.append_issues("{}. {}".format(i, issue), '#e74c3c')
        else:
            self.append_issues("No major issues found!\n", '#27ae60')

        self.append_issues("")

        if recommendations:
            self.append_issues("RECOMMENDATIONS ({}):\n".format(len(recommendations)), '#3498db')
            for i, rec in enumerate(recommendations, 1):
                self.append_issues("{}. {}".format(i, rec), '#3498db')
        else:
            self.append_issues("Keep up the good work!\n", '#27ae60')

        self.append_issues("\n" + "=" * 60)

        # Feature highlights
        self.append_issues("\nKEY METRICS:", '#f39c12')
        self.append_issues("-" * 60)
        self.append_issues("Functions: {:.0f}".format(features.get('num_functions', 0)))
        self.append_issues("Classes: {:.0f}".format(features.get('num_classes', 0)))
        self.append_issues("Avg Function Length: {:.1f} lines".format(features.get('avg_function_length', 0)))
        self.append_issues("Cyclomatic Complexity: {:.0f}".format(features.get('cyclomatic_complexity', 0)))
        self.append_issues("Docstring Coverage: {:.1f}%".format(features.get('docstring_ratio', 0) * 100))
        self.append_issues("Type Hint Coverage: {:.1f}%".format(features.get('type_hint_ratio', 0) * 100))
        self.append_issues("Error Handling: {:.1f}%".format(features.get('error_handling_ratio', 0) * 100))
        self.append_issues("=" * 60)

    def analyze_repository(self):
        """Start repository analysis in background thread"""
        repo_url = self.url_entry.get().strip()

        if not repo_url or repo_url == "https://github.com/username/repository":
            messagebox.showwarning("Input Error", "Please enter a valid GitHub repository URL")
            return

        if self.is_analyzing:
            messagebox.showinfo("Busy", "Analysis already in progress")
            return

        # Disable buttons
        self.analyze_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.terminal_output.delete(1.0, tk.END)
        self.result_label.config(text="Analyzing...", bg='#f39c12', fg='white')
        self.progress.start(10)
        self.is_analyzing = True

        # Run in background thread
        thread = threading.Thread(target=self.run_analysis, args=(repo_url,))
        thread.daemon = True
        thread.start()

    def clone_repository(self, repo_url, repo_name):
        """Clone repository using git command"""
        repo_path = self.clone_dir / repo_name

        # Remove if already exists
        if repo_path.exists():
            self.append_output(f"Removing existing directory: {repo_name}")
            shutil.rmtree(repo_path)

        # Clone repository
        self.append_output(f"Cloning repository: {repo_url}")
        try:
            result = subprocess.run(
                ['git', 'clone', repo_url, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                raise Exception(f"Git clone failed: {result.stderr}")

            self.append_output(f"Successfully cloned to: {repo_path}")
            return repo_path

        except subprocess.TimeoutExpired:
            raise Exception("Clone operation timed out (2 minutes)")
        except FileNotFoundError:
            raise Exception("Git is not installed. Please install git first.")

    def extract_features_from_local(self, repo_path):
        """Extract features from local repository"""
        self.append_output(f"Scanning Python files in: {repo_path}")

        python_files = []
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env', 'node_modules']]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        self.append_output(f"Found {len(python_files)} Python files")

        if not python_files:
            raise Exception("No Python files found in repository")

        # Limit to first 20 files for analysis
        python_files = python_files[:20]
        self.append_output(f"Analyzing {len(python_files)} Python files")

        all_features = []
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_content = f.read()

                analyzer = CodeAnalyzer(code_content)
                features = analyzer.extract_all_features()
                all_features.append(features)

            except Exception as e:
                self.append_output(f"Skipping {os.path.basename(file_path)}: {str(e)}")
                continue

        if not all_features:
            raise Exception("Failed to extract features from any Python file")

        # Average features across all files
        import numpy as np
        aggregated = {}
        feature_keys = all_features[0].keys()

        for key in feature_keys:
            values = [f[key] for f in all_features if key in f]
            aggregated[key] = np.mean(values) if values else 0

        self.append_output(f"Successfully extracted {len(aggregated)} features")
        return aggregated

    def run_analysis(self, repo_url):
        """Run analysis in background"""
        repo_path = None
        try:
            self.append_output("=" * 70)
            self.append_output("Starting analysis...")
            self.append_output("=" * 70)

            # Load model
            self.append_output("\nLoading model...")
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)

            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.append_output(f"Model loaded: {metadata['best_model_name']}")
            self.append_output(f"Model accuracy: {metadata['test_accuracy']:.4f}")

            # Clone repository
            self.append_output("\n" + "=" * 70)
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = self.clone_repository(repo_url, repo_name)

            # Extract features from local repository
            self.append_output("\n" + "=" * 70)
            self.append_output("Extracting features...")
            features = self.extract_features_from_local(repo_path)

            # Prepare and predict
            self.append_output("\n" + "=" * 70)
            self.append_output("Running prediction...")
            feature_df = pd.DataFrame([features])

            for feat in metadata['feature_names']:
                if feat not in feature_df.columns:
                    feature_df[feat] = 0

            feature_df = feature_df[metadata['feature_names']]

            # Normalize features while keeping DataFrame structure
            features_normalized = pd.DataFrame(
                scaler.transform(feature_df),
                columns=metadata['feature_names']
            )

            prediction = model.predict(features_normalized)[0]

            # Get confidence (quality score percentage)
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features_normalized)[0]
                confidence = max(probability) * 100
            else:
                confidence = None

            # Result
            result = {
                'repo_name': repo_name,
                'repo_url': repo_url,
                'prediction': int(prediction),
                'quality': 'HIGH QUALITY' if prediction == 1 else 'LOW QUALITY',
                'confidence': confidence,
                'quality_score': confidence if prediction == 1 else (100 - confidence) if confidence else None,
                'features': features,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': metadata['best_model_name']
            }

            self.current_result = result
            self.append_output("\n" + "=" * 70)
            self.append_output("Analysis complete!")
            self.append_output("=" * 70)

            # Update UI on main thread
            self.root.after(0, self.display_result, result)

        except Exception as e:
            error_msg = str(e)
            self.append_output(f"\nERROR: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", error_msg))
            self.root.after(0, self.analysis_finished)

        finally:
            # Clean up cloned repository
            if repo_path and repo_path.exists():
                self.append_output(f"\nCleaning up: Removing {repo_path}")
                try:
                    shutil.rmtree(repo_path)
                except Exception as e:
                    self.append_output(f"Warning: Could not remove temp directory: {e}")

    def display_result(self, result):
        """Display prediction result"""
        quality = result['quality']
        confidence = result.get('confidence')
        quality_score = result.get('quality_score')

        if quality == 'HIGH QUALITY':
            color = '#27ae60'
            emoji = 'PASS'
        else:
            color = '#e74c3c'
            emoji = 'FAIL'

        result_text = f"{emoji} {quality}"
        if quality_score:
            result_text += f" (Score: {quality_score:.1f}/100)"

        self.result_label.config(text=result_text, bg=color, fg='white')

        self.append_output("\n" + "=" * 70)
        self.append_output(f"Repository: {result['repo_name']}")
        self.append_output(f"URL: {result['repo_url']}")
        self.append_output(f"Quality: {quality}")
        if quality_score:
            self.append_output(f"Quality Score: {quality_score:.2f}/100")
        if confidence:
            self.append_output(f"Model Confidence: {confidence:.2f}%")
        self.append_output(f"Model Used: {result['model_name']}")
        self.append_output("=" * 70)

        # Analyze quality issues
        self.analyze_quality_issues(result['features'], quality_score if quality_score else 50)

        self.save_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        self.analysis_finished()

    def analysis_finished(self):
        """Re-enable buttons after analysis"""
        self.progress.stop()
        self.analyze_btn.config(state=tk.NORMAL)
        self.is_analyzing = False

    def save_result(self):
        """Save prediction result"""
        if not self.current_result:
            return

        try:
            # Create directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            repo_safe_name = self.current_result['repo_name'].replace('/', '_')
            test_dir = Path('../test_results') / f"{timestamp}_{repo_safe_name}"
            test_dir.mkdir(parents=True, exist_ok=True)

            # Save files
            with open(test_dir / 'result.json', 'w') as f:
                json.dump(self.current_result, f, indent=2)

            with open(test_dir / 'report.txt', 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("CODE QUALITY PREDICTION REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Repository: {self.current_result['repo_name']}\n")
                f.write(f"URL: {self.current_result['repo_url']}\n")
                f.write(f"Test Date: {self.current_result['timestamp']}\n\n")
                f.write(f"Quality: {self.current_result['quality']}\n")
                if self.current_result['confidence']:
                    f.write(f"Confidence: {self.current_result['confidence']:.2f}%\n")
                f.write(f"\nModel: {self.current_result['model_name']}\n")

            features_df = pd.DataFrame([self.current_result['features']])
            features_df.to_csv(test_dir / 'features.csv', index=False)

            self.append_output(f"\nResults saved to: {test_dir}")
            messagebox.showinfo("Success", f"Results saved to:\n{test_dir}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {str(e)}")

    def show_help(self):
        """Show help dialog"""
        help_text = """
CODE QUALITY PREDICTOR - HELP

HOW TO USE:
1. Enter a GitHub repository URL (e.g., https://github.com/user/repo)
2. Click 'Analyze Repository' to start analysis
3. View results in Terminal Output and Quality Issues sections
4. Click 'Save Result' to save analysis to file
5. Click 'Export Features' to export feature data as CSV

QUALITY SCORING:
- Score 70-100: High Quality (PASS)
- Score 50-69: Moderate Quality
- Score 0-49: Low Quality (FAIL)

The quality score is based on:
- Documentation (docstrings, comments)
- Type hints usage
- Error handling
- Code complexity
- Naming conventions
- Function/class structure

MODEL USED:
- SVM (RBF Kernel) classifier
- Trained on 165 GitHub repositories
- 22 extracted code quality features

For issues or questions, contact the ML Coursework team.
        """
        messagebox.showinfo("Help - Code Quality Predictor", help_text)

    def export_features(self):
        """Export features to CSV"""
        if not self.current_result:
            return

        try:
            from tkinter import filedialog

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_name = f"features_{self.current_result['repo_name']}_{timestamp}.csv"

            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if file_path:
                features_df = pd.DataFrame([self.current_result['features']])
                features_df.to_csv(file_path, index=False)
                self.append_output(f"\nFeatures exported to: {file_path}")
                messagebox.showinfo("Success", f"Features exported to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    def clear_all(self):
        """Clear all inputs and outputs"""
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, "https://github.com/username/repository")
        self.terminal_output.delete(1.0, tk.END)
        self.issues_output.delete(1.0, tk.END)
        self.issues_output.insert(tk.END, "Quality analysis will appear here after repository analysis...")
        self.result_label.config(text="No analysis yet", bg='#ecf0f1', fg='#2c3e50')
        self.progress.stop()
        self.save_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.current_result = None


def main():
    """Main execution"""
    root = tk.Tk()
    app = CodeQualityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
