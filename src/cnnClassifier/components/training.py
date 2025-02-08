from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None  # Initialize model as None



        # Remove the create_model method
        self.model = self.get_base_model()  # Use the model from prepare_base_model



    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

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
        model.save(path)  # Saving in the .keras format

    def train(self, callback_list: list):
        # Calculate steps per epoch and validation steps
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # Save the model after training
        self.save_model(path=self.config.trained_model_path, model=self.model)

        # Return the training history
        return history

# Execute the training
if __name__ == "__main__":
    config = TrainingConfig()  # Assuming you have a way to initialize this
    training_instance = Training(config)
    training_instance.train_valid_generator()
    training_instance.train(callback_list=[])  # Pass any callbacks if needed
