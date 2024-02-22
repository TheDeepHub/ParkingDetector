# app.py
from flask import Flask, request, jsonify
import tensorflow as tf  # Use `import onnxruntime as ort` for an ONNX model

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('path_to_my_model.h5')  # Adjust for ONNX if necessary

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess your input data here
    prediction = model.predict(data['input'])
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
