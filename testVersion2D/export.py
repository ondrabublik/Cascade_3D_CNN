import tensorflow as tf
from pathlib import Path

# Cesty
base_path = Path('../../data/training_data/test_2D')
keras_model_path = base_path / 'model.keras'
tflite_model_path = base_path / 'model.tflite'

# Načti Keras model
model = tf.keras.models.load_model(keras_model_path)

# Vytvoř TFLite converter z Keras modelu
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Ulož TFLite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model uložen do: {tflite_model_path}")
