import mlflow.keras
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
from flask import Flask, request, jsonify
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
char_to_num = None
num_to_char = None
max_len = None
image_width = None
image_height = None

def get_latest_model_version(model_name="HandwritingRecognitionModel"):
    """Retrieve the latest available model version."""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            logger.error(f"No registered model found with name: {model_name}")
            return None
        
        latest_version = max(int(v.version) for v in versions)
        logger.info(f"Latest model version: {latest_version}")
        return latest_version
    except Exception as e:
        logger.error(f"Error fetching model version: {e}")
        return None

def load_model(model_uri):
    """Load the MLflow model and preprocessing artifacts."""
    global model, char_to_num, num_to_char, max_len, image_width, image_height

    try:
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.keras.load_model(model_uri)
        
        # Load model config
        artifact_path = os.path.dirname(model_uri)
        config_path = os.path.join(artifact_path, "artifacts", "model_config.json")
        
        if not os.path.exists(config_path):
            logger.warning(f"Model config not found at {config_path}, using default parameters")
            return False

        with open(config_path, "r") as f:
            model_config = json.load(f)

        # Extract parameters
        characters = model_config["characters"]
        max_len = model_config["max_len"]
        image_width = model_config["image_width"]
        image_height = model_config["image_height"]

        # Create character mappings
        char_to_num = StringLookup(vocabulary=characters, mask_token=None)
        num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

        logger.info("Model and artifacts loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def distortion_free_resize(image, img_size):
    """Resize images without distortion."""
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Compute padding
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    pad_height_top = pad_height // 2
    pad_height_bottom = pad_height - pad_height_top
    pad_width_left = pad_width // 2
    pad_width_right = pad_width - pad_width_left

    image = tf.pad(image, [[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])

    # Match training data transformation
    image = tf.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image

def process_image(image_data):
    """Process image for model input."""
    try:
        img = tf.io.decode_image(image_data, channels=1)
        img = distortion_free_resize(img, (image_width, image_height))
        img = tf.cast(img, tf.float32) / 255.0
        return tf.expand_dims(img, 0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def decode_predictions(pred):
    """Decode model predictions to text."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)

    return output_text[0] if output_text else ""

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for handwriting recognition."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        image_data = image_file.read()

        processed_image = process_image(image_data)
        predictions = model.predict(processed_image)
        recognized_text = decode_predictions(predictions)

        return jsonify({"recognized_text": recognized_text})

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

def start_service(model_name="HandwritingRecognitionModel", model_version=None, port=5000):
    """Start the API service with the specified model."""
    # Get latest version if not provided
    if model_version is None:
        model_version = get_latest_model_version(model_name)
        if model_version is None:
            logger.error("No valid model version found. Exiting.")
            return

    model_uri = f"models:/{model_name}/{model_version}"
    
    if not load_model(model_uri):
        logger.error("Failed to load model. Service cannot start.")
        return

    logger.info(f"Starting API service on port {port}")
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.info("No model version provided. Fetching the latest version...")
        model_version = get_latest_model_version("HandwritingRecognitionModel")
        if model_version is None:
            print("No valid model version found. Exiting.")
            sys.exit(1)
    else:
        model_version = sys.argv[1]


    model_version = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    start_service(model_version=model_version, port=port)
