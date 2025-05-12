import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)

class ChestXRayModel:
    def __init__(self):
        # Check for model in both docker and local paths
        docker_path = '/app/data/models/chest_xray_model.h5'
        local_path = 'data/models/chest_xray_model.h5'
        
        if os.path.exists(docker_path):
            self.model_path = docker_path
            logger.info(f"Using model at Docker path: {docker_path}")
        else:
            self.model_path = local_path
            logger.info(f"Using model at local path: {local_path}")
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        self.img_size = (224, 224)
        self.num_classes = 4  # Normal, COVID, Pneumonia, Opacity
        self.class_names = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
        
        # Load or create the model
        if os.path.exists(self.model_path):
            logger.info(f"Loading model from {self.model_path}")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.error(traceback.format_exc())
                logger.info("Creating new model instead")
                self.model = self._create_model()
        else:
            logger.info(f"Model file not found at {self.model_path}, creating new model")
            self.model = self._create_model()
    
    def _create_model(self):
        """Create a transfer learning model based on DenseNet121."""
        try:
            # Load pre-trained DenseNet121 without the top classification layer
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
            
            # Freeze the base model layers
            for layer in base_model.layers:
                layer.trainable = False
            
            # Add custom classification layers
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            
            # Create the model
            model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            logger.error(traceback.format_exc())
            
            # Create a simple fallback model
            inputs = tf.keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
            x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
    
    def train(self, train_dir, validation_dir, epochs=10, batch_size=32):
        """Train the model on chest X-ray images."""
        # Create data generators for augmentation and preprocessing
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators from directories
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Update class names from the generator
        self.class_names = list(train_generator.class_indices.keys())
        logger.info(f"Classes found: {self.class_names}")
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        # Save the trained model
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return history
    
    def predict(self, img_array):
        """Make predictions on a preprocessed image."""
        # Ensure the image is in the right format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Add batch dimension if not present
            img_batch = np.expand_dims(img_array, axis=0)
            
            try:
                # Make prediction
                logger.info("Running model prediction")
                predictions = self.model.predict(img_batch)[0]
                logger.info(f"Prediction results: {predictions}")
                
                # Process the results
                results = {
                    "primary_finding": self.class_names[np.argmax(predictions)],
                    "confidence": float(np.max(predictions)),
                    "all_findings": [
                        {"condition": class_name, "confidence": float(conf)}
                        for class_name, conf in zip(self.class_names, predictions)
                    ]
                }
                
                logger.info(f"Primary finding: {results['primary_finding']} with confidence {results['confidence']}")
                return results
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                logger.error(traceback.format_exc())
                # Return a fallback result if prediction fails
                return {
                    "primary_finding": "Unknown",
                    "confidence": 0.0,
                    "all_findings": [{"condition": "Error", "confidence": 0.0}],
                    "error": str(e)
                }
        else:
            raise ValueError(f"Expected RGB image with shape (224, 224, 3), got {img_array.shape}")
    
    def evaluate(self, test_dir, batch_size=32):
        """Evaluate the model on a test dataset."""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate
        results = self.model.evaluate(test_generator)
        
        logger.info(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
        
        return results