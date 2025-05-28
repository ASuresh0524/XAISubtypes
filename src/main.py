"""
Main script for breast cancer subtype classification
"""

import os
import argparse
import logging
from datetime import datetime

from data.data_processor import DataProcessor
from models.cnn_model import BreastCancerCNN
from models.tree_model import BreastCancerTreeModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_cnn(data_dir, output_dir, epochs=50, batch_size=32):
    """Train the CNN model"""
    logger.info("Preparing data for CNN training...")
    processor = DataProcessor(data_dir)
    X_train, y_train, X_val, y_val = processor.prepare_data()
    
    logger.info("Initializing CNN model...")
    model = BreastCancerCNN(num_classes=y_train.shape[1])
    model.compile_model()
    
    logger.info("Training CNN model...")
    history = model.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"cnn_model_{timestamp}")
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, history

def train_tree_model(data_dir, output_dir, model_type='xgboost'):
    """Train the tree-based model"""
    logger.info(f"Preparing data for {model_type} training...")
    processor = DataProcessor(data_dir)
    X_train, y_train, X_val, y_val = processor.prepare_data()
    
    # Flatten the images for tree-based models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    logger.info(f"Initializing {model_type} model...")
    model = BreastCancerTreeModel(model_type=model_type)
    
    logger.info("Training tree model...")
    model.train(X_train_flat, y_train, X_val_flat, y_val)
    
    # Calculate feature importance
    importance = model.get_feature_importance()
    logger.info("Feature importance calculated")
    
    # Calculate SHAP values for interpretability
    logger.info("Calculating SHAP values...")
    shap_values = model.calculate_shap_values(X_train_flat[:100])  # Use subset for SHAP
    
    return model, importance, shap_values

def main():
    parser = argparse.ArgumentParser(description="Train breast cancer subtype classification models")
    parser.add_argument("--data_dir", required=True, help="Directory containing the image data")
    parser.add_argument("--output_dir", required=True, help="Directory to save model outputs")
    parser.add_argument("--model_type", choices=['cnn', 'xgboost', 'lightgbm'], default='cnn',
                        help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for CNN training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CNN training")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model_type == 'cnn':
        model, history = train_cnn(
            args.data_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        model, importance, shap_values = train_tree_model(
            args.data_dir,
            args.output_dir,
            model_type=args.model_type
        )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 