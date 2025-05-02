import os
from torchvision.datasets import CIFAR10
from PIL import Image
import logging
from logging_utilis import setup_logging

setup_logging("download_cifar.log")

def download_cifar():
    """Download CIFAR-10 dataset and organize into class folders"""
    base_path = os.path.abspath("data/raw")
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    try:
        logging.info("Downloading CIFAR-10 dataset...")
        # Create root directory
        os.makedirs(base_path, exist_ok=True)
        logging.info(f"Created directory: {base_path}")

        # Download dataset
        train_dataset = CIFAR10(root=base_path, train=True, download=True)
        test_dataset = CIFAR10(root=base_path, train=False, download=True)

        # Process both train and test sets
        for dataset, prefix in [(train_dataset, 'train'), (test_dataset, 'test')]:
            for idx, (image, label) in enumerate(dataset):
                class_dir = os.path.join(base_path, classes[label])
                os.makedirs(class_dir, exist_ok=True)
                
                # Save image as PNG
                img_path = os.path.join(class_dir, f"{prefix}_{idx}.png")
                image.save(img_path)

        logging.info("Dataset successfully processed and organized")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    download_cifar()