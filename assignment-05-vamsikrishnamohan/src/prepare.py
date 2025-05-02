import os
import yaml
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import logging
from logging_utilis import setup_logging

setup_logging("Data Prepation.log")


def prepare_data():
    try:
        logging.info("Starting data preparation process...")
        
        # Load configuration
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        version = params['data']['version']
        seed = params['seed']
        
        # Define source and destination paths
        source_dir = "data/current"
        processed_dir = f"data/processed/{version}"
        
        # Remove existing processed data
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        
        # Create train, val, and test directories
        os.makedirs(f"{processed_dir}/train", exist_ok=True)
        os.makedirs(f"{processed_dir}/val", exist_ok=True)
        os.makedirs(f"{processed_dir}/test", exist_ok=True)
        
        # Split data into train, val, and test sets
        for class_name in os.listdir(source_dir):
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # List all files in the class directory
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
            
            # Split into train+val and test sets
            train_val_files, test_files = train_test_split(files, test_size=0.2, random_state=seed)
            
            # Split train+val into train and val sets
            train_files, val_files = train_test_split(train_val_files, test_size=0.25, random_state=seed)
            
            # Copy files to respective directories
            for file in train_files:
                dest_dir = os.path.join(processed_dir, "train", class_name)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(file, dest_dir)
            
            for file in val_files:
                dest_dir = os.path.join(processed_dir, "val", class_name)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(file, dest_dir)
            
            for file in test_files:
                dest_dir = os.path.join(processed_dir, "test", class_name)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(file, dest_dir)
        
        logging.info(f"Data preparation for version {version} completed successfully.")
    
    except Exception as e:
        logging.error(f"Data preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    prepare_data()