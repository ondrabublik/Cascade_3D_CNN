import base64
from pathlib import Path

base_path = Path('../../data/training_data/test_2D')

input_file = base_path / 'model.tflite'
output_file =  base_path / 'model_base64.txt'

with open(input_file, "rb") as f:
    model_bytes = f.read()

# Převedení na Base64
model_b64 = base64.b64encode(model_bytes).decode("utf-8")

# Uložení jako text
with open(output_file, "w") as f:
    f.write(model_b64)

print("✅ Hotovo! Base64 uložen do:", output_file)
