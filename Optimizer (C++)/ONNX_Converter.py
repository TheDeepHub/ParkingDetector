import tensorflow as tf
import tf2onnx

# Load your Keras model
model = tf.keras.models.load_model('path_to_my_model.h5')

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13, output_path="model.onnx")
