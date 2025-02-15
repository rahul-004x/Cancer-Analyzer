from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        self.trained_models = {}
        self.results = {}
        
    def train_models(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
    def get_feature_importance(self, feature_names):
        if 'Random Forest' in self.trained_models:
            rf_model = self.trained_models['Random Forest']
            importance = rf_model.feature_importances_
            return dict(zip(feature_names, importance))
        return None
    
    def predict(self, features):
        predictions = {}
        for name, model in self.trained_models.items():
            pred_proba = model.predict_proba(features)[0]
            predictions[name] = {
                'prediction': 'Malignant' if model.predict(features)[0] else 'Benign',
                'probability': max(pred_proba)
            }
        return predictions
