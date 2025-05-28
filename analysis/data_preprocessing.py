"""
Data Preprocessing and Validation for Breast Cancer Subtype Analysis

This script provides utilities for data preprocessing, validation, and quality control
for the breast cancer subtype classification project.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir='../data', output_dir='../data/processed'):
        """
        Initialize the data preprocessor
        
        Args:
            data_dir (str): Directory containing the raw data
            output_dir (str): Directory for saving processed data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize scaler
        self.scaler = StandardScaler()
    
    def validate_data_format(self, df):
        """
        Validate the format of input data
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            dict: Dictionary of validation results
        """
        checks = {
            "Has Breast_Cancer_Subtype column": "Breast_Cancer_Subtype" in df.columns,
            "All gene values are numeric": df.drop(columns=["Breast_Cancer_Subtype"]).dtypes.apply(
                lambda x: np.issubdtype(x, np.number)
            ).all(),
            "No missing values": not df.isnull().any().any(),
            "Proper shape": df.shape[1] > 1  # At least one feature column plus target
        }
        
        for check, result in checks.items():
            logger.info(f"{check}: {'✓' if result else '✗'}")
        
        return checks
    
    def preprocess_data(self, df, log2_transform=True, standardize=True):
        """
        Preprocess gene expression data
        
        Args:
            df (pd.DataFrame): Input dataset
            log2_transform (bool): Whether to apply log2 transformation
            standardize (bool): Whether to apply standardization
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Store target variable
        target = processed_df["Breast_Cancer_Subtype"]
        gene_data = processed_df.drop(columns=["Breast_Cancer_Subtype"])
        
        # Log2 transformation
        if log2_transform:
            logger.info("Applying log2 transformation")
            gene_data = np.log2(gene_data + 1)  # Add 1 to handle zeros
        
        # Standardization
        if standardize:
            logger.info("Applying standardization")
            gene_data = pd.DataFrame(
                self.scaler.fit_transform(gene_data),
                columns=gene_data.columns,
                index=gene_data.index
            )
        
        # Combine processed data with target
        processed_df = pd.concat([gene_data, target], axis=1)
        
        return processed_df
    
    def analyze_data_distribution(self, df):
        """
        Analyze the distribution of data and subtypes
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            tuple: (subtype_distribution, gene_statistics)
        """
        # Subtype distribution
        subtype_dist = df["Breast_Cancer_Subtype"].value_counts()
        logger.info("\nSubtype Distribution:")
        logger.info(subtype_dist)
        
        # Basic statistics of gene expression values
        gene_stats = df.drop(columns=["Breast_Cancer_Subtype"]).describe()
        logger.info("\nGene Expression Statistics:")
        logger.info(gene_stats)
        
        return subtype_dist, gene_stats
    
    def remove_low_variance_genes(self, df, threshold=0.01):
        """
        Remove genes with low variance
        
        Args:
            df (pd.DataFrame): Input dataset
            threshold (float): Variance threshold
            
        Returns:
            pd.DataFrame: Dataset with low variance genes removed
        """
        # Calculate variances
        gene_vars = df.drop(columns=["Breast_Cancer_Subtype"]).var()
        
        # Get genes above threshold
        high_var_genes = gene_vars[gene_vars > threshold].index
        
        # Filter dataset
        filtered_df = df[list(high_var_genes) + ["Breast_Cancer_Subtype"]]
        
        logger.info(f"Removed {len(gene_vars) - len(high_var_genes)} low variance genes")
        return filtered_df
    
    def create_directory_structure(self):
        """Create the required directory structure"""
        directories = [
            os.path.join(self.data_dir, "raw/expression"),
            os.path.join(self.data_dir, "raw/metadata"),
            os.path.join(self.data_dir, "processed/train"),
            os.path.join(self.data_dir, "processed/test"),
            os.path.join(self.data_dir, "results/models"),
            os.path.join(self.data_dir, "results/figures")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def process_and_validate(self, input_file, output_file=None):
        """
        Complete data processing pipeline
        
        Args:
            input_file (str): Path to input file
            output_file (str): Path to save processed data (optional)
        
        Returns:
            pd.DataFrame: Processed dataset
        """
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Validate format
        validation_results = self.validate_data_format(df)
        if not all(validation_results.values()):
            logger.warning("Data validation failed!")
            return None
        
        # Analyze distribution
        self.analyze_data_distribution(df)
        
        # Remove low variance genes
        df = self.remove_low_variance_genes(df)
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Save processed data
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            processed_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        
        return processed_df

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    preprocessor.create_directory_structure()
    
    # Process data
    # processed_data = preprocessor.process_and_validate(
    #     "path_to_your_data.csv",
    #     "processed_data.csv"
    # ) 