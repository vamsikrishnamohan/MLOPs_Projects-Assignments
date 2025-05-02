import yaml
import os
import shutil
import logging
import time
from logging_utilis import setup_logging

# Initialize logging
setup_logging("data_loader.log")

def load_data():
    """Pull data version from DVC and prepare current dataset"""
    try:
        logging.info("Starting data loading process...")
        
        # Load configuration
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
            
        version = params['data']['version'].strip()  # Remove leading/trailing spaces
        seed = params['seed']
        output_dir = os.path.abspath("data/current")
        
        logging.info(f"Loading dataset version: {version} with seed {seed}")
        
        # Remove existing data
        if os.path.exists(output_dir):
            logging.info(f"Removing existing data in {output_dir}...")
            retries = 3  # Number of retries
            for i in range(retries):
                try:
                    shutil.rmtree(output_dir)
                    logging.info(f"Successfully removed {output_dir}.")
                    break
                except Exception as e:
                    logging.warning(f"Attempt {i + 1}: Failed to remove {output_dir}. Retrying...")
                    if i == retries - 1:
                        raise e
                    time.sleep(1)  # Wait before retrying
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle combined versions (v1+v2, v1+v2+v3)
        if "+" in version:
            # Combine data from multiple partitions
            parts = version.split("+")
            for part in parts:
                part = part.strip()  # Remove spaces from each part
                part_path = os.path.abspath(f"data/{part}")
                if not os.path.exists(part_path):
                    raise FileNotFoundError(f"Dataset partition {part} not found at {part_path}")
                
                # Copy data from each partition
                for class_dir in os.listdir(part_path):
                    class_path = os.path.join(part_path, class_dir)
                    if os.path.isdir(class_path):
                        dest_class_path = os.path.join(output_dir, class_dir)
                        os.makedirs(dest_class_path, exist_ok=True)
                        for img in os.listdir(class_path):
                            src_img = os.path.join(class_path, img)
                            dest_img = os.path.join(dest_class_path, img)
                            shutil.copy(src_img, dest_img)
            
            logging.info(f"Successfully combined data from {version} to {output_dir}")
        else:
            # Handle single version (v1, v2, v3)
            data_path = os.path.abspath(f"data/{version}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset version {version} not found at {data_path}")
            
            # Copy data from the single partition
            logging.info(f"Copying data from {data_path} to {output_dir}...")
            shutil.copytree(data_path, output_dir)
            logging.info(f"Successfully loaded {version} to {output_dir}")

    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        raise

if __name__ == "__main__":
    load_data()