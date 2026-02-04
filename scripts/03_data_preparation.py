"""
Phase 3: Data Preparation
Clean, normalize, and prepare data for ML model training.

Author: Manikanta Gonugondla
Co-Author: Claude Sonnet 4.5
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os


class DataPreparator:
    """Prepare extracted features for ML training"""

    def __init__(self, input_file='../data/extracted_features.csv'):
        self.df = pd.read_csv(input_file)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def analyze_data(self):
        """Analyze dataset characteristics"""
        print("=" * 70)
        print("Dataset Analysis")
        print("=" * 70)

        print(f"\nDataset shape: {self.df.shape}")
        print(f"Total samples: {len(self.df)}")
        print(f"Total features: {len(self.df.columns) - 2}")  # Exclude repo_name and quality_label

        # Class distribution
        print("\nClass Distribution:")
        print(self.df['quality_label'].value_counts())
        print(f"\nClass balance:")
        print(self.df['quality_label'].value_counts(normalize=True) * 100)

        # Missing values
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found")
        else:
            print(missing[missing > 0])

        # Basic statistics
        print("\nFeature Statistics (first 5 features):")
        print(self.df.iloc[:, :5].describe())

    def handle_missing_values(self):
        """Handle any missing values"""
        print("\n" + "=" * 70)
        print("Handling Missing Values")
        print("=" * 70)

        # Fill missing values with 0 (assuming missing means feature doesn't exist)
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'quality_label']

        for col in numeric_columns:
            if self.df[col].isnull().any():
                self.df[col].fillna(0, inplace=True)
                print(f"Filled {col} missing values with 0")

        print("Missing value handling complete")

    def remove_outliers(self):
        """Remove extreme outliers using IQR method"""
        print("\n" + "=" * 70)
        print("Outlier Detection & Removal")
        print("=" * 70)

        initial_count = len(self.df)

        # Get feature columns (exclude repo_name and quality_label)
        feature_cols = [col for col in self.df.columns if col not in ['repo_name', 'quality_label']]

        # Calculate IQR for each feature
        Q1 = self.df[feature_cols].quantile(0.25)
        Q3 = self.df[feature_cols].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds (using 3*IQR for less aggressive removal)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Remove rows with outliers in ANY feature
        mask = ~((self.df[feature_cols] < lower_bound) | (self.df[feature_cols] > upper_bound)).any(axis=1)
        self.df = self.df[mask]

        removed = initial_count - len(self.df)
        print(f"Removed {removed} outlier samples ({removed/initial_count*100:.1f}%)")
        print(f"Remaining samples: {len(self.df)}")

    def normalize_features(self):
        """Normalize features using StandardScaler"""
        print("\n" + "=" * 70)
        print("Feature Normalization")
        print("=" * 70)

        # Separate features from labels
        self.feature_columns = [col for col in self.df.columns if col not in ['repo_name', 'quality_label']]

        X = self.df[self.feature_columns]
        y = self.df['quality_label']

        # Fit and transform
        X_normalized = self.scaler.fit_transform(X)

        # Create normalized dataframe
        df_normalized = pd.DataFrame(X_normalized, columns=self.feature_columns, index=self.df.index)
        df_normalized['repo_name'] = self.df['repo_name'].values
        df_normalized['quality_label'] = self.df['quality_label'].values

        self.df = df_normalized

        print("Features normalized using StandardScaler (mean=0, std=1)")
        print(f"Normalized feature range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")

    def create_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n" + "=" * 70)
        print("Train-Test Split")
        print("=" * 70)

        X = self.df[self.feature_columns]
        y = self.df['quality_label']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Train set size: {len(self.X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Test set size: {len(self.X_test)} ({test_size*100:.0f}%)")

        print("\nTrain set class distribution:")
        print(self.y_train.value_counts())

        print("\nTest set class distribution:")
        print(self.y_test.value_counts())

    def visualize_features(self):
        """Create visualizations of feature distributions"""
        print("\n" + "=" * 70)
        print("Creating Feature Visualizations")
        print("=" * 70)

        # Create output directory
        os.makedirs('../evaluation', exist_ok=True)

        # 1. Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[self.feature_columns].corr()
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('../evaluation/feature_correlation.png', dpi=150)
        print("Saved: feature_correlation.png")
        plt.close()

        # 2. Feature distributions by class
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()

        for idx, col in enumerate(self.feature_columns[:9]):  # First 9 features
            for label in [0, 1]:
                data = self.df[self.df['quality_label'] == label][col]
                axes[idx].hist(data, alpha=0.6, label=f'Label {label}', bins=20)
            axes[idx].set_title(col)
            axes[idx].legend()
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('../evaluation/feature_distributions.png', dpi=150)
        print("Saved: feature_distributions.png")
        plt.close()

        # 3. Class balance visualization
        plt.figure(figsize=(8, 6))
        self.df['quality_label'].value_counts().plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
        plt.title('Class Distribution')
        plt.xlabel('Quality Label (0=Low, 1=High)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('../evaluation/class_distribution.png', dpi=150)
        print("Saved: class_distribution.png")
        plt.close()

    def save_prepared_data(self):
        """Save prepared datasets"""
        print("\n" + "=" * 70)
        print("Saving Prepared Data")
        print("=" * 70)

        # Save train and test sets
        train_df = pd.DataFrame(self.X_train, columns=self.feature_columns)
        train_df['quality_label'] = self.y_train.values
        train_df.to_csv('../data/train_set.csv', index=False)
        print("Saved: train_set.csv")

        test_df = pd.DataFrame(self.X_test, columns=self.feature_columns)
        test_df['quality_label'] = self.y_test.values
        test_df.to_csv('../data/test_set.csv', index=False)
        print("Saved: test_set.csv")

        # Save full normalized dataset
        self.df.to_csv('../data/normalized_features.csv', index=False)
        print("Saved: normalized_features.csv")

        # Save scaler for future use
        import pickle
        with open('../models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("Saved: scaler.pkl")


def main():
    """Main execution"""
    print("=" * 70)
    print("Phase 3: Data Preparation & Preprocessing")
    print("=" * 70)

    # Initialize preparator
    preparator = DataPreparator()

    # Step 1: Analyze data
    preparator.analyze_data()

    # Step 2: Handle missing values
    preparator.handle_missing_values()

    # Step 3: Remove outliers
    preparator.remove_outliers()

    # Step 4: Normalize features
    preparator.normalize_features()

    # Step 5: Create train-test split
    preparator.create_train_test_split(test_size=0.2, random_state=42)

    # Step 6: Visualize features
    preparator.visualize_features()

    # Step 7: Save prepared data
    preparator.save_prepared_data()

    print("\n" + "=" * 70)
    print("Phase 3 Complete - Data Preparation Successful!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("- data/train_set.csv (training data)")
    print("- data/test_set.csv (testing data)")
    print("- data/normalized_features.csv (full normalized dataset)")
    print("- models/scaler.pkl (feature scaler)")
    print("- evaluation/feature_correlation.png")
    print("- evaluation/feature_distributions.png")
    print("- evaluation/class_distribution.png")

    print("\nNext Steps:")
    print("1. Review generated visualizations in evaluation/")
    print("2. Check train/test split files")
    print("3. Move to Phase 4: Model Training")


if __name__ == "__main__":
    main()
