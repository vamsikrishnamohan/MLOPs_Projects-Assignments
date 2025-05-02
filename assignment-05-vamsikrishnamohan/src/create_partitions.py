import os
import random
import shutil
import logging
from logging_utilis import setup_logging

# Initialize logging
setup_logging("create_partitions.log")

def create_partitions():
    """Create dataset partitions v1, v2, v3, v1+v2, and v1+v2+v3"""
    try:
        logging.info("Starting partition creation process...")
        
        # Define source and destination directories
        source_dir = os.path.abspath("data/raw")
        dest_base = os.path.abspath("data")
        
        # Define partitions
        partitions = {
            "v1": 20000,  # 20k samples
            "v2": 20000,  # 20k samples
            "v3": 20000,  # 20k samples
            "v1+v2": 40000,  # 40k samples (v1 + v2)
            "v1+v2+v3": 60000  # 60k samples (v1 + v2 + v3)
        }
        
        # Collect all images (skip non-image files)
        all_images = []
        for class_dir in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_dir)
            if os.path.isdir(class_path):  # Skip files like .tar.gz
                all_images += [os.path.join(class_path, img) 
                             for img in os.listdir(class_path) 
                             if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Verify total images
        if len(all_images) < 60000:
            raise ValueError(f"Expected 60k images, got {len(all_images)}")

        # Shuffle and split
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(all_images)
        
        # Create partitions
        for part, samples in partitions.items():
            part_dir = os.path.join(dest_base, part)
            os.makedirs(part_dir, exist_ok=True)
            
            # For combined partitions (v1+v2, v1+v2+v3), copy images from existing partitions
            if "+" in part:
                logging.info(f"Creating combined partition: {part}")
                for sub_part in part.split("+"):
                    sub_part_dir = os.path.join(dest_base, sub_part)
                    if not os.path.exists(sub_part_dir):
                        raise FileNotFoundError(f"Partition {sub_part} not found at {sub_part_dir}")
                    
                    # Copy images from sub-partition to combined partition
                    for class_dir in os.listdir(sub_part_dir):
                        class_path = os.path.join(sub_part_dir, class_dir)
                        if os.path.isdir(class_path):
                            dest_class_path = os.path.join(part_dir, class_dir)
                            os.makedirs(dest_class_path, exist_ok=True)
                            for img in os.listdir(class_path):
                                src_img = os.path.join(class_path, img)
                                dest_img = os.path.join(dest_class_path, img)
                                shutil.copy(src_img, dest_img)
                
                logging.info(f"Created combined partition {part} with images from {part.split('+')}")
            else:
                # For single partitions (v1, v2, v3), create from the shuffled list
                part_images = all_images[:samples]  # Take the first N samples
                all_images = all_images[samples:]  # Remove used samples
                
                # Copy images
                for img_path in part_images:
                    cls = os.path.basename(os.path.dirname(img_path))
                    dest = os.path.join(part_dir, cls, os.path.basename(img_path))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy(img_path, dest)
                
                logging.info(f"Created partition {part} with {len(part_images)} images")

        logging.info("Partition creation completed successfully.")

    except Exception as e:
        logging.error(f"Partitioning failed: {str(e)}")
        raise

if __name__ == "__main__":
    create_partitions()