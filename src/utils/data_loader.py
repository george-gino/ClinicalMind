import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil

def download_covid_dataset(output_dir="data/datasets/covid_xray"):
    """
    Download the COVID-19 Radiography Database from Kaggle and organize it.
    
    This dataset contains chest X-rays for:
    - COVID-19 positive cases
    - Normal cases
    - Lung opacity (non-COVID pneumonia)
    - Viral pneumonia
    
    Returns:
        DataFrame with metadata about the images
    """
    print(f"Downloading COVID-19 X-ray dataset to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download the metadata
        metadata = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "tawsifurrahman/covid19-radiography-database",
            "COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/metadata.csv"
        )
        
        print(f"Downloaded metadata with {len(metadata)} records")
        
        # Download some sample images (adjust numbers based on your needs)
        sample_size = 50  # Number of images per class to download
        
        # Define the classes
        classes = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
        
        # Create directories for each class
        for class_name in classes:
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
        
        # Sample and download images for each class
        for class_name in classes:
            print(f"Downloading {sample_size} images for class: {class_name}")
            
            # Filter metadata for this class
            class_metadata = metadata[metadata['Label'] == class_name]
            
            # Take a sample
            if len(class_metadata) > sample_size:
                class_metadata = class_metadata.sample(sample_size)
            
            # Download each image
            for idx, row in class_metadata.iterrows():
                try:
                    img_path = row['File Path']
                    img_data = kagglehub.load_dataset(
                        KaggleDatasetAdapter.FILE,
                        "tawsifurrahman/covid19-radiography-database",
                        img_path
                    )
                    
                    # Copy to appropriate directory
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(output_dir, class_name, filename)
                    shutil.copy(img_data, dest_path)
                    
                except Exception as e:
                    print(f"Error downloading {img_path}: {e}")
        
        print(f"Dataset downloaded to {output_dir}")
        return metadata
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def prepare_dataset_splits(dataset_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    """
    Organize the dataset into train/validation/test splits.
    
    Args:
        dataset_dir: Directory containing class subdirectories with images
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
    """
    import random
    
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
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
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
        
        print(f"Class {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    print(f"Dataset split into train/val/test directories")

def visualize_samples(dataset_dir, num_samples=5):
    """
    Visualize sample images from each class in the dataset.
    
    Args:
        dataset_dir: Directory containing class subdirectories with images
        num_samples: Number of samples to visualize per class
    """
    # Get all class directories
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    # Configure the plot
    fig, axes = plt.subplots(len(class_dirs), num_samples, figsize=(num_samples*3, len(class_dirs)*3))
    
    # For each class
    for i, class_name in enumerate(class_dirs):
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Select a random sample
        if len(images) > num_samples:
            import random
            images = random.sample(images, num_samples)
        
        # Display each image
        for j, img_name in enumerate(images):
            if j < num_samples:
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                
                # Display in the appropriate subplot
                ax = axes[i, j] if len(class_dirs) > 1 else axes[j]
                ax.imshow(img, cmap='gray')
                ax.set_title(f"{class_name}")
                ax.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(os.path.dirname(dataset_dir), "sample_visualization.png")
    plt.savefig(output_path)
    print(f"Sample visualization saved to {output_path}")
    
    return fig

if __name__ == "__main__":
    # Example usage
    metadata = download_covid_dataset()
    prepare_dataset_splits("data/datasets/covid_xray")
    visualize_samples("data/datasets/covid_xray")