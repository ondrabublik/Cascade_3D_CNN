from pathlib import Path
import tensorflow as tf
print(tf.__version__)
import tensorflowjs as tfjs


# Cesty
base_path = Path('../../data/training_data/test_2D')
keras_model_path = base_path / 'model.keras'
tflite_model_path = base_path / 'model.tflite'

# Načti Keras model
model = tf.keras.models.load_model(keras_model_path)

# převod na TFJS formát
tfjs.converters.save_keras_model(model, base_path / 'web_model')

