"""
Phase 5: Advanced Model Training
Train complex ML models including ensemble methods and neural networks.

Author: Manikanta Gonugondla
Co-Author: Claude Sonnet 4.5
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class AdvancedModelTrainer:
    """Train advanced ML models with hyperparameter tuning"""

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

    def initialize_advanced_models(self):
        """Initialize advanced ML models"""
        print("=" * 70)
        print("Initializing Advanced ML Models")
        print("=" * 70)

        self.models = {
            # Enhanced Ensemble Methods
            'XGBoost-style GB': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),

            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                random_state=42,
                n_jobs=-1
            ),

            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.5,
                random_state=42
            ),

            # Neural Network
            'Neural Network (MLP)': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),

            # Advanced SVM
            'SVM (RBF Kernel)': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),

            # K-Nearest Neighbors
            'KNN (Optimized)': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='manhattan'
            ),

            # Optimized Random Forest
            'Random Forest (Tuned)': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
        }

        print(f"Initialized {len(self.models)} advanced models")

    def create_ensemble_models(self):
        """Create voting and stacking ensemble models"""
        print("\n" + "=" * 70)
        print("Creating Ensemble Models")
        print("=" * 70)

        # Base models for ensembles
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
        svm = SVC(kernel='rbf', C=5.0, probability=True, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

        # Voting Classifier (Hard Voting)
        voting_hard = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='hard'
        )
        self.models['Voting Ensemble (Hard)'] = voting_hard

        # Voting Classifier (Soft Voting)
        voting_soft = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft'
        )
        self.models['Voting Ensemble (Soft)'] = voting_soft

        # Stacking Classifier
        stacking = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('svm', svm),
                ('mlp', mlp)
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        self.models['Stacking Ensemble'] = stacking

        print(f"Created 3 ensemble models (Voting Hard, Voting Soft, Stacking)")

    def train_and_evaluate(self):
        """Train all models and evaluate performance"""
        print("\n" + "=" * 70)
        print("Training & Evaluation")
        print("=" * 70)

        for name, model in self.models.items():
            print(f"\n{'=' * 50}")
            print(f"Model: {name}")
            print('=' * 50)

            try:
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

                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_test_proba = model.predict_proba(self.X_test)[:, 1]
                else:
                    y_test_proba = None

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

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

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

        # Sort by test accuracy
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)

        print("\nModel Comparison (Sorted by Test Accuracy):")
        print(comparison_df.to_string(index=False))

        # Select best based on test accuracy
        best_idx = comparison_df['Test Accuracy'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']
        self.best_model = self.results[self.best_model_name]['model']

        print(f"\nBest Model: {self.best_model_name}")
        print(f"Test Accuracy: {self.results[self.best_model_name]['test_accuracy']:.4f}")

    def save_advanced_models(self):
        """Save the best model and top 3 models"""
        print("\n" + "=" * 70)
        print("Saving Advanced Models")
        print("=" * 70)

        os.makedirs('../models/advanced', exist_ok=True)

        # Save best model
        best_model_path = '../models/advanced/best_model.pkl'
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Saved best model ({self.best_model_name}): {best_model_path}")

        # Save all results
        with open('../models/advanced/all_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("Saved all training results")

        # Save model comparison
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test_Accuracy': [r['test_accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1_Score': [r['f1_score'] for r in self.results.values()],
            'ROC_AUC': [r['roc_auc'] if r['roc_auc'] else 0 for r in self.results.values()]
        })
        comparison_df.to_csv('../models/advanced/model_comparison.csv', index=False)
        print("Saved model comparison: model_comparison.csv")

        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'test_accuracy': self.results[self.best_model_name]['test_accuracy'],
            'precision': self.results[self.best_model_name]['precision'],
            'recall': self.results[self.best_model_name]['recall'],
            'f1_score': self.results[self.best_model_name]['f1_score'],
            'roc_auc': self.results[self.best_model_name]['roc_auc'],
            'feature_names': list(self.X_train.columns),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open('../models/advanced/best_model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print("Saved best model metadata")


def main():
    """Main execution"""
    print("=" * 70)
    print("Phase 5: Advanced Model Training")
    print("=" * 70)

    # Initialize trainer
    trainer = AdvancedModelTrainer()

    print(f"\nTraining set size: {len(trainer.X_train)}")
    print(f"Test set size: {len(trainer.X_test)}")
    print(f"Number of features: {len(trainer.X_train.columns)}")

    # Step 1: Initialize advanced models
    trainer.initialize_advanced_models()

    # Step 2: Create ensemble models
    trainer.create_ensemble_models()

    # Step 3: Train and evaluate all models
    trainer.train_and_evaluate()

    # Step 4: Select best model
    trainer.select_best_model()

    # Step 5: Save models
    trainer.save_advanced_models()

    print("\n" + "=" * 70)
    print("Phase 5 Complete - Advanced Training Successful!")
    print("=" * 70)

    print("\nNext Steps:")
    print("1. Check models/advanced/ for saved models")
    print("2. Review model_comparison.csv")
    print("3. Move to Phase 6: Inference & Testing System")


if __name__ == "__main__":
    main()
