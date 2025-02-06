import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None  # Initialize model as None

    def get_base_model(self):
        """Load the VGG16 model and save it as the base model."""
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        # Save the initial base model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """Customize the model by adding a dense layer and setting layers' trainability."""
        # Set trainability of layers based on freeze_all and freeze_till
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add custom dense layer for classification
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        # Define the full model with modified layers
        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        # Compile with the new configurations (use Adam optimizer by default)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        full_model.summary()  # Optionally print summary for debugging
        return full_model

    def update_base_model(self):
        """Update the base model with custom configurations and save it."""
        # Ensure base model is loaded
        if self.model is None:
            self.get_base_model()
        
        # Update the model with a custom output layer and training configuration
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the model at the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        model.save(path)