#!/usr/bin/env python
import os
import argparse
import logging
import matplotlib.pyplot as plt
from src.models.deep_learning_model import ChestXRayModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_training_history(history, output_path='data/models/training_history.png'):
    """Plot and save the training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Training history plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train chest X-ray model')
    parser.add_argument('--train_dir', required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', required=True, help='Path to validation data directory')
    parser.add_argument('--test_dir', help='Path to test data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    logger.info(f"Initializing model...")
    model = ChestXRayModel()
    
    logger.info(f"Training model on {args.train_dir} for {args.epochs} epochs...")
    history = model.train(
        args.train_dir, 
        args.val_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    # Plot training history
    if history is not None:
        plot_training_history(history)
    
    # Evaluate on test set if provided
    if args.test_dir:
        logger.info(f"Evaluating model on {args.test_dir}...")
        results = model.evaluate(args.test_dir, batch_size=args.batch_size)
        logger.info(f"Test results: Loss={results[0]:.4f}, Accuracy={results[1]:.4f}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()