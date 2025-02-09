"""
CNN Model Training Module for Chicken Disease Classification

This module implements the training pipeline for the Chicken Disease Classification model.
It handles data preprocessing, model creation, training, and performance visualization.
The implementation uses transfer learning with VGG16 as the base model and adds custom
layers for the specific classification task.

Key Features:
- Transfer learning using VGG16
- Data augmentation for training
- Early stopping and learning rate scheduling
- Performance visualization
- Model checkpointing
"""

# Third-party imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

class Training:
    """
    Handles the training process for the Chicken Disease Classification model.
    
    This class manages the entire training pipeline, including:
    - Model initialization and compilation
    - Data preprocessing and augmentation
    - Training process with callbacks
    - Model saving and performance visualization
    
    Attributes:
        config: Configuration object containing training parameters
        model: The neural network model (VGG16 based)
        train_generator: Generator for training data
        valid_generator: Generator for validation data
    """

    def __init__(self, config):
        """
        Initialize the training process with given configuration.
        
        Args:
            config: Configuration object containing training parameters
        """
        self.config = config
        self.model = self.get_base_model()  # Initialize the model
        self.compile_model()  # Compile the model
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        """
        Create and return the model architecture.
        
        Returns:
            tf.keras.Model: Compiled model with VGG16 base and custom top layers
        """
        # Initialize VGG16 base model
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.config.params_image_size
        )

        # Add custom classification layers
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(4, activation='softmax')(x)

        return tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    def compile_model(self):
        """
        Compile the model with optimizer, loss function, and metrics.
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        """
        Set up data generators for training and validation.
        
        Configures data preprocessing and augmentation pipelines:
        - Rescales pixel values to [0,1]
        - Applies data augmentation for training (if enabled)
        - Splits data into training and validation sets
        """
        # Common parameters for data generators
        datagenerator_kwargs = dict(rescale=1./255, validation_split=0.20)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Setup validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode="categorical",
            **dataflow_kwargs
        )

        # Setup training generator with optional augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode="categorical",
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model in multiple formats.
        
        Args:
            path (Path): Path to save the model
            model (tf.keras.Model): Trained model to save
        """
        model.save(path)  # Save in .keras format
        import joblib
        joblib.dump(model, path.with_suffix('.pkl'))  # Save as pickle file

    def train(self, callback_list):
        """
        Train the model with the configured parameters.
        
        Implements the training loop with:
        - Class weight balancing
        - Early stopping
        - Learning rate scheduling
        - Model checkpointing
        - History tracking
        
        Returns:
            dict: Training history containing metrics
        """
        if not self.train_generator or not self.valid_generator:
            print("Error: Generators are not initialized. Run train_valid_generator() first.")
            return

        # Calculate class weights for imbalanced dataset
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_generator.classes),
            y=self.train_generator.classes
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Setup training callbacks
        callback_list = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                self.config.trained_model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
                min_lr=self.config.lr_scheduler_min_lr,
                verbose=1
            )
        ]

        # Calculate steps per epoch
        steps_per_epoch = max(1, self.train_generator.samples // self.train_generator.batch_size)
        validation_steps = max(1, self.valid_generator.samples // self.valid_generator.batch_size)

        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list,
        )

        # Save model and training artifacts
        self.save_model(path=self.config.trained_model_path, model=self.model)

        # Prepare history data
        history_data = {
            'epochs': list(range(1, len(history.history['accuracy']) + 1)),
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }

        # Save training history
        history_path = Path("static/history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=4)

        # Generate and save performance plots
        self.save_plots(history_data)

        return history_data

    def save_plots(self, history_data):
        """
        Generate and save training performance visualization plots.
        
        Args:
            history_data (dict): Training history metrics
        """
        # Plot accuracy curves
        plt.figure(figsize=(10, 5))
        plt.plot(history_data['epochs'], history_data['accuracy'], label='Training Accuracy')
        plt.plot(history_data['epochs'], history_data['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.savefig("static/accuracy_vs_val_accuracy.png")

        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(history_data['epochs'], history_data['loss'], label='Training Loss')
        plt.plot(history_data['epochs'], history_data['val_loss'], label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.savefig("static/loss_vs_val_loss.png")
