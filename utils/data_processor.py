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
        """Load data from a CSV file and perform initial cleaning"""
        try:
            print("Loading data from:", file_path)
            self.data = pd.read_csv(file_path)
            return self._clean_data()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def load_data_from_df(self, df):
        """Load data from a pandas DataFrame and perform initial cleaning"""
        try:
            print("Loading data from DataFrame")
            self.data = df.copy()
            return self._clean_data()
        except Exception as e:
            print(f"Error loading DataFrame: {str(e)}")
            return None

    def _clean_data(self):
        """Clean the loaded data"""
        if self.data is None:
            return None

        try:
            # Remove unnamed columns
            unnamed_cols = [col for col in self.data.columns if 'Unnamed' in col]
            if unnamed_cols:
                self.data = self.data.drop(columns=unnamed_cols)

            # Verify required columns
            required_cols = ['diagnosis']
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError("Missing required columns in the dataset")

            # Initialize features for visualization
            self.X = self.data.drop(['id', 'diagnosis'], axis=1)
            self.y = (self.data['diagnosis'] == 'M').astype(int)

            print("Data shape after cleaning:", self.data.shape)
            return self.data

        except Exception as e:
            print(f"Error cleaning data: {str(e)}")
            self.data = None
            return None

    def preprocess_data(self):
        """Preprocess data including handling missing values"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        print("Starting data preprocessing...")
        try:
            # Handle missing values
            print("Handling missing values...")
            print("Shape before imputation:", self.X.shape)
            X_imputed = self.imputer.fit_transform(self.X)

            # Scale features
            print("Scaling features...")
            X_scaled = self.scaler.fit_transform(X_imputed)
            self.X = pd.DataFrame(X_scaled, columns=self.X.columns)

            # Split data
            print("Splitting data into train and test sets...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            print("Data preprocessing completed.")
            print(f"Training set shape: {self.X_train.shape}")
            print(f"Test set shape: {self.X_test.shape}")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            raise

    def get_feature_stats(self):
        """Get basic statistics of the dataset"""
        if self.data is not None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            return self.data[numeric_cols].describe()
        return None

    def get_correlation_matrix(self):
        """Get correlation matrix of features"""
        if self.X is not None:
            try:
                return self.X.corr()
            except Exception as e:
                print(f"Error calculating correlation matrix: {str(e)}")
                return None
        return None

    def validate_input_data(self, input_data):
        """Validate input data for predictions"""
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if not all(col in input_data.columns for col in self.X.columns):
            raise ValueError("Input data missing required features")

        return True

    def get_feature_names(self):
        """Get list of feature names"""
        return list(self.X.columns) if self.X is not None else []