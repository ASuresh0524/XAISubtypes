"""
Gradient Boosted Tree model implementation for breast cancer subtype classification
"""

import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

class BreastCancerTreeModel:
    def __init__(self, model_type='xgboost', random_state=42):
        """
        Initialize the tree model
        
        Args:
            model_type (str): Type of model to use ('xgboost' or 'lightgbm')
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'lightgbm':
            self.model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state
            )
        else:
            raise ValueError("model_type must be either 'xgboost' or 'lightgbm'")
        
        self.feature_importance = None
        self.shap_values = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def calculate_shap_values(self, X):
        """Calculate SHAP values for model interpretability"""
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)
        return self.shap_values
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.feature_importance is None:
            raise ValueError("Model must be trained before getting feature importance")
        return self.feature_importance 