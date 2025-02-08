import tensorflow as tf
import matplotlib.pyplot as plt
import json
from pathlib import Path

class Training:
    def __init__(self, config):
        self.config = config
        self.model = self.get_base_model()
        self.compile_model()

    def get_base_model(self):
        base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=self.config.params_image_size)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(4, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        return model

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1./255, validation_split=0.20)
        dataflow_kwargs = dict(target_size=self.config.params_image_size[:-1], batch_size=self.config.params_batch_size, interpolation="bilinear")

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(directory=self.config.training_data, subset="validation", shuffle=False, **dataflow_kwargs)

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(directory=self.config.training_data, subset="training", shuffle=True, class_mode="categorical", **dataflow_kwargs)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)

        history_data = {
            'epochs': list(range(1, len(history.history['accuracy']) + 1)),
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }

        # Save history data as JSON
        history_path = Path("static/history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=4)

        # Save plots
        self.save_plots(history_data)
        return history_data

    def save_plots(self, history_data):
        plt.figure(figsize=(10, 5))
        plt.plot(history_data['epochs'], history_data['accuracy'], label='Training Accuracy')
        plt.plot(history_data['epochs'], history_data['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.savefig("static/accuracy_vs_val_accuracy.png")

        plt.figure(figsize=(10, 5))
        plt.plot(history_data['epochs'], history_data['loss'], label='Training Loss')
        plt.plot(history_data['epochs'], history_data['val_loss'], label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.savefig("static/loss_vs_val_loss.png")
