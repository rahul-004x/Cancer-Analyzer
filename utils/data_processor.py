import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def load_data(self, file_path):
        """Load data and perform initial cleaning"""
        print("Loading data from:", file_path)
        self.data = pd.read_csv(file_path)

        # Remove unnamed columns
        unnamed_cols = [col for col in self.data.columns if 'Unnamed' in col]
        if unnamed_cols:
            self.data = self.data.drop(columns=unnamed_cols)

        print("Data shape after cleaning:", self.data.shape)
        return self.data

    def preprocess_data(self):
        """Preprocess data including handling missing values"""
        print("Starting data preprocessing...")

        # Drop ID column and convert diagnosis to binary
        X = self.data.drop(['id', 'diagnosis'], axis=1)
        y = (self.data['diagnosis'] == 'M').astype(int)

        # Handle missing values
        print("Handling missing values...")
        print("Shape before imputation:", X.shape)
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.X = pd.DataFrame(X_scaled, columns=X.columns)
        self.y = y

        # Split data
        print("Splitting data into train and test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        print("Data preprocessing completed.")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_stats(self):
        """Get basic statistics of the dataset"""
        return self.data.describe()

    def get_correlation_matrix(self):
        """Get correlation matrix of features"""
        return self.X.corr() if self.X is not None else None