"""
Phase 4: Model Training & Evaluation
Train multiple ML models and select the best performer.

Author: Manikanta Gonugondla
Co-Author: Claude Sonnet 4.5
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ModelTrainer:
    """Train and evaluate multiple ML models"""

    def __init__(self):
        # Load prepared data
        self.X_train = pd.read_csv('../data/train_set.csv').drop('quality_label', axis=1)
        self.y_train = pd.read_csv('../data/train_set.csv')['quality_label']
        self.X_test = pd.read_csv('../data/test_set.csv').drop('quality_label', axis=1)
        self.y_test = pd.read_csv('../data/test_set.csv')['quality_label']

        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def initialize_models(self):
        """Initialize ML models with default parameters"""
        print("=" * 70)
        print("Initializing ML Models")
        print("=" * 70)

        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }

        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")

    def train_and_evaluate(self):
        """Train all models and evaluate performance"""
        print("\n" + "=" * 70)
        print("Training & Evaluation")
        print("=" * 70)

        for name, model in self.models.items():
            print(f"\n{'=' * 50}")
            print(f"Model: {name}")
            print('=' * 50)

            # Cross-validation
            print("Performing 5-fold cross-validation...")
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Train on full training set
            print("Training on full training set...")
            model.fit(self.X_train, self.y_train)

            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            y_test_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            train_acc = accuracy_score(self.y_train, y_train_pred)
            test_acc = accuracy_score(self.y_test, y_test_pred)
            precision = precision_score(self.y_test, y_test_pred)
            recall = recall_score(self.y_test, y_test_pred)
            f1 = f1_score(self.y_test, y_test_pred)

            # ROC-AUC
            if y_test_proba is not None:
                roc_auc = roc_auc_score(self.y_test, y_test_proba)
            else:
                roc_auc = None

            # Store results
            self.results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba
            }

            # Print results
            print(f"\nResults:")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  Precision:      {precision:.4f}")
            print(f"  Recall:         {recall:.4f}")
            print(f"  F1-Score:       {f1:.4f}")
            if roc_auc:
                print(f"  ROC-AUC:        {roc_auc:.4f}")

    def select_best_model(self):
        """Select best model based on test accuracy"""
        print("\n" + "=" * 70)
        print("Model Selection")
        print("=" * 70)

        # Compare models
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'CV Mean': [r['cv_mean'] for r in self.results.values()],
            'Test Accuracy': [r['test_accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()]
        })

        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))

        # Select best based on test accuracy
        best_idx = comparison_df['Test Accuracy'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']
        self.best_model = self.results[self.best_model_name]['model']

        print(f"\nBest Model: {self.best_model_name}")
        print(f"Test Accuracy: {self.results[self.best_model_name]['test_accuracy']:.4f}")

    def generate_evaluation_report(self):
        """Generate detailed evaluation report for best model"""
        print("\n" + "=" * 70)
        print(f"Detailed Evaluation Report - {self.best_model_name}")
        print("=" * 70)

        y_pred = self.results[self.best_model_name]['y_test_pred']

        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Low Quality', 'High Quality']))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Save metrics to file
        os.makedirs('../evaluation', exist_ok=True)
        with open('../evaluation/metrics.txt', 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"Code Quality Prediction Model - Evaluation Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Best Model: {self.best_model_name}\n\n")

            f.write("Model Comparison:\n")
            f.write("-" * 70 + "\n")
            for name, results in self.results.items():
                f.write(f"\n{name}:\n")
                f.write(f"  CV Accuracy:    {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})\n")
                f.write(f"  Test Accuracy:  {results['test_accuracy']:.4f}\n")
                f.write(f"  Precision:      {results['precision']:.4f}\n")
                f.write(f"  Recall:         {results['recall']:.4f}\n")
                f.write(f"  F1-Score:       {results['f1_score']:.4f}\n")
                if results['roc_auc']:
                    f.write(f"  ROC-AUC:        {results['roc_auc']:.4f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Best Model Details: {self.best_model_name}\n")
            f.write("=" * 70 + "\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(self.y_test, y_pred, target_names=['Low Quality', 'High Quality']))

        print("\nSaved metrics to: evaluation/metrics.txt")

    def create_visualizations(self):
        """Create evaluation visualizations"""
        print("\n" + "=" * 70)
        print("Creating Visualizations")
        print("=" * 70)

        # 1. Model Comparison Bar Chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        models_list = list(self.results.keys())
        test_acc = [self.results[m]['test_accuracy'] for m in models_list]
        f1_scores = [self.results[m]['f1_score'] for m in models_list]

        axes[0].bar(models_list, test_acc, color='#4ecdc4')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Test Accuracy Comparison')
        axes[0].set_ylim([0.5, 1.0])
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].bar(models_list, f1_scores, color='#ff6b6b')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('F1-Score Comparison')
        axes[1].set_ylim([0.5, 1.0])
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('../evaluation/model_comparison.png', dpi=150)
        print("Saved: model_comparison.png")
        plt.close()

        # 2. Confusion Matrix for Best Model
        y_pred = self.results[self.best_model_name]['y_test_pred']
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low Quality', 'High Quality'],
                    yticklabels=['Low Quality', 'High Quality'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('../evaluation/confusion_matrix.png', dpi=150)
        print("Saved: confusion_matrix.png")
        plt.close()

        # 3. ROC Curve
        plt.figure(figsize=(8, 6))
        for name, results in self.results.items():
            if results['y_test_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results['y_test_proba'])
                auc = results['roc_auc']
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('../evaluation/roc_curve.png', dpi=150)
        print("Saved: roc_curve.png")
        plt.close()

        # 4. Feature Importance (for tree-based models)
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            importances = self.best_model.feature_importances_
            feature_names = self.X_train.columns
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], color='#95e1d3')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importances - {self.best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('../evaluation/feature_importance.png', dpi=150)
            print("Saved: feature_importance.png")
            plt.close()

    def save_best_model(self):
        """Save the best model to disk"""
        print("\n" + "=" * 70)
        print("Saving Best Model")
        print("=" * 70)

        os.makedirs('../models', exist_ok=True)

        # Save model
        model_path = '../models/quality_classifier.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)

        print(f"Saved best model ({self.best_model_name}) to: models/quality_classifier.pkl")

        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'test_accuracy': self.results[self.best_model_name]['test_accuracy'],
            'precision': self.results[self.best_model_name]['precision'],
            'recall': self.results[self.best_model_name]['recall'],
            'f1_score': self.results[self.best_model_name]['f1_score'],
            'roc_auc': self.results[self.best_model_name]['roc_auc'],
            'feature_names': list(self.X_train.columns)
        }

        with open('../models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print("Saved model metadata to: models/model_metadata.pkl")


def main():
    """Main execution"""
    print("=" * 70)
    print("Phase 4: Model Training & Evaluation")
    print("=" * 70)

    # Initialize trainer
    trainer = ModelTrainer()

    print(f"\nTraining set size: {len(trainer.X_train)}")
    print(f"Test set size: {len(trainer.X_test)}")
    print(f"Number of features: {len(trainer.X_train.columns)}")

    # Step 1: Initialize models
    trainer.initialize_models()

    # Step 2: Train and evaluate all models
    trainer.train_and_evaluate()

    # Step 3: Select best model
    trainer.select_best_model()

    # Step 4: Generate detailed report
    trainer.generate_evaluation_report()

    # Step 5: Create visualizations
    trainer.create_visualizations()

    # Step 6: Save best model
    trainer.save_best_model()

    print("\n" + "=" * 70)
    print("Phase 4 Complete - Model Training Successful!")
    print("=" * 70)

    print("\nGenerated Files:")
    print("- models/quality_classifier.pkl (trained model)")
    print("- models/model_metadata.pkl (model info)")
    print("- evaluation/metrics.txt (detailed metrics)")
    print("- evaluation/model_comparison.png")
    print("- evaluation/confusion_matrix.png")
    print("- evaluation/roc_curve.png")
    print("- evaluation/feature_importance.png")

    print("\nNext Steps:")
    print("1. Review evaluation metrics in evaluation/metrics.txt")
    print("2. Analyze visualizations in evaluation/")
    print("3. Test model predictions on new repositories")
    print("4. Move to Phase 5: Report Writing")


if __name__ == "__main__":
    main()
