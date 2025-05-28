"""
Breast Cancer Subtype Analysis using XGBoost and SHAP

This script implements breast cancer subtype classification using XGBoost
and provides interpretability through SHAP (SHapley Additive exPlanations) values.
"""

import os
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold

# Configure matplotlib for better visualization
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)

class BreastCancerAnalysis:
    def __init__(self, data_dir='../data/raw', output_dir='../outputs', model_dir='../models'):
        """
        Initialize the analysis pipeline
        
        Args:
            data_dir (str): Directory containing the raw data
            output_dir (str): Directory for saving outputs
            model_dir (str): Directory for saving models
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_dir = model_dir
        
        # Create directories if they don't exist
        for directory in [data_dir, output_dir, model_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def load_data(self, file_path):
        """
        Load and prepare the gene expression data
        
        Args:
            file_path (str): Path to the gene expression data file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    
    def split_data(self, df, target_column="Breast_Cancer_Subtype", n_splits=5):
        """
        Split data using StratifiedKFold
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of the target column
            n_splits (int): Number of folds
            
        Returns:
            list: List of (train, test) splits
        """
        skf = StratifiedKFold(n_splits=n_splits)
        target = df[target_column]
        
        splits = []
        for fold_no, (train_index, test_index) in enumerate(skf.split(df, target), 1):
            train = df.loc[train_index, :]
            test = df.loc[test_index, :]
            
            # Save splits
            train_filename = os.path.join(self.output_dir, f'train_split_{fold_no}.csv')
            test_filename = os.path.join(self.output_dir, f'test_split_{fold_no}.csv')
            train.to_csv(train_filename, index=False)
            test.to_csv(test_filename, index=False)
            
            splits.append((train, test))
            print(f"Fold {fold_no}: Train shape {train.shape}, Test shape {test.shape}")
        
        return splits
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, fold_no=None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            fold_no: Fold number for saving model
            
        Returns:
            xgb.XGBRFClassifier: Trained model
        """
        model = xgb.XGBRFClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        
        if fold_no is not None:
            model_path = os.path.join(self.model_dir, f'model_fold_{fold_no}.json')
            model.save_model(model_path)
        
        return model
    
    def analyze_shap_values(self, model, X_test, save_plot=True, fold_no=None):
        """
        Calculate and analyze SHAP values
        
        Args:
            model: Trained model
            X_test: Test data
            save_plot (bool): Whether to save the plot
            fold_no: Fold number for saving plot
            
        Returns:
            tuple: (shap_values, explainer)
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Plot SHAP summary
        plt.figure(figsize=(15, 10))
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.title("Feature Importance Based on SHAP Values")
        
        if save_plot and fold_no is not None:
            plot_path = os.path.join(self.output_dir, f'shap_summary_fold_{fold_no}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        
        plt.close()
        return shap_values, explainer
    
    def analyze_patient_specific_genes(self, shap_values, feature_names, n_top_genes=20):
        """
        Analyze top genes for each patient
        
        Args:
            shap_values: SHAP values
            feature_names: Names of the features/genes
            n_top_genes (int): Number of top genes to analyze
            
        Returns:
            pd.DataFrame: DataFrame with top genes for each patient
        """
        patient_genes = []
        
        for patient_idx in range(len(shap_values)):
            # Get absolute SHAP values for this patient
            patient_shap = np.abs(shap_values[patient_idx])
            
            # Get indices of top genes
            top_indices = np.argsort(patient_shap)[-n_top_genes:]
            
            # Get gene names and their SHAP values
            top_genes = {
                feature_names[i]: shap_values[patient_idx][i]
                for i in top_indices
            }
            
            patient_genes.append(top_genes)
        
        return pd.DataFrame(patient_genes)
    
    def run_analysis(self, data_file):
        """
        Run the complete analysis pipeline
        
        Args:
            data_file (str): Path to the input data file
        """
        # Load data
        df = self.load_data(data_file)
        
        # Split data
        splits = self.split_data(df)
        
        # Process each fold
        for fold_no, (train, test) in enumerate(splits, 1):
            print(f"\nProcessing fold {fold_no}")
            
            # Prepare data
            X_train = train.drop(columns=["Breast_Cancer_Subtype"])
            y_train = train["Breast_Cancer_Subtype"]
            X_test = test.drop(columns=["Breast_Cancer_Subtype"])
            y_test = test["Breast_Cancer_Subtype"]
            
            # Train model
            model = self.train_model(X_train, y_train, X_test, y_test, fold_no)
            
            # SHAP analysis
            shap_values, _ = self.analyze_shap_values(model, X_test, True, fold_no)
            
            # Patient-specific gene analysis
            patient_genes = self.analyze_patient_specific_genes(
                shap_values, X_test.columns
            )
            
            # Save patient-specific genes
            output_file = os.path.join(
                self.output_dir, f'patient_specific_genes_fold_{fold_no}.csv'
            )
            patient_genes.to_csv(output_file)
            
            print(f"Completed analysis for fold {fold_no}")

if __name__ == "__main__":
    # Example usage
    analyzer = BreastCancerAnalysis()
    # analyzer.run_analysis("path_to_your_data.csv") 