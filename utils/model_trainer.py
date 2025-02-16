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
        """Train and evaluate models"""
        print("\nStarting model training...")

        for name, model in self.models.items():
            try:
                print(f"\nTraining {name}...")
                # Check for NaN values using DataFrame.isna() method
                if X_train.isna().any().any():
                    raise ValueError("Training data contains NaN values")

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

                print(f"{name} training completed.")
                print(f"Accuracy: {self.results[name]['accuracy']:.4f}")

            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue

    def get_feature_importance(self, feature_names):
        """Get feature importance for Random Forest model"""
        if 'Random Forest' in self.trained_models:
            rf_model = self.trained_models['Random Forest']
            importance = rf_model.feature_importances_
            return dict(zip(feature_names, importance))
        return None

    def predict(self, features):
        """Make predictions using trained models"""
        predictions = {}
        for name, model in self.trained_models.items():
            try:
                pred_proba = model.predict_proba(features)[0]
                predictions[name] = {
                    'prediction': 'Malignant' if model.predict(features)[0] else 'Benign',
                    'probability': float(max(pred_proba))
                }
            except Exception as e:
                print(f"Error making prediction with {name}: {str(e)}")
                continue
        return predictions