import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
        
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self.data
    
    def preprocess_data(self):
        # Drop ID column and convert diagnosis to binary
        X = self.data.drop(['id', 'diagnosis'], axis=1)
        y = (self.data['diagnosis'] == 'M').astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.X = pd.DataFrame(X_scaled, columns=X.columns)
        self.y = y
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_stats(self):
        return self.data.describe()
    
    def get_correlation_matrix(self):
        return self.X.corr()
