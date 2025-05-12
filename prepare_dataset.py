#!/usr/bin/env python3
import os
import sys
import random
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_dataset_splits(dataset_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    """Organize the dataset into train/validation/test splits."""
    # Create output directories
    output_dir = os.path.dirname(dataset_dir)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process each class
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for class_name in class_dirs:
        # Create class directories in each split
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all images for this class
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle for randomization
        random.shuffle(images)
        
        # Calculate split indices
        train_end = int(len(images) * train_split)
        val_end = train_end + int(len(images) * val_split)
        
        # Split the images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Copy images to their respective directories
        for img_list, target_dir in [
            (train_images, os.path.join(train_dir, class_name)),
            (val_images, os.path.join(val_dir, class_name)),
            (test_images, os.path.join(test_dir, class_name))
        ]:
            for img in img_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(target_dir, img)
                shutil.copy2(src, dst)
        
        logger.info(f"Class {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    logger.info(f"Dataset split into train/val/test directories")

# Main execution
dataset_dir = "data/datasets/covid_xray"
prepare_dataset_splits(dataset_dir)

logger.info("Dataset preparation complete!")
logger.info("You can now train your model with:")
logger.info(f"python3 train_model.py --train_dir data/datasets/train --val_dir data/datasets/val --epochs 5")