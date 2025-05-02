import mlflow
import mlflow.sklearn
import mlflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import StringLookup
from keras import ops
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import sys
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)

def main(split_ratio=(0.9, 0.05, 0.05)):
    """
    Main function to run the handwriting recognition model with MLflow tracking
    
    Args:
        split_ratio (tuple): Ratio for train/validation/test split (train, val, test)
    """
    try:
        # Enable MLflow autologging for Keras
        mlflow.keras.autolog()
        
        # Set Experiment Name
        mlflow.set_experiment("Handwriting_Recognition")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log the split ratio as parameters
            mlflow.log_param("train_split", split_ratio[0])
            mlflow.log_param("validation_split", split_ratio[1])
            mlflow.log_param("test_split", split_ratio[2])
            
            logger.info(f"Starting experiment with split ratio: {split_ratio}")
            
            # Load and prepare the data
            train_ds, validation_ds, test_ds, char_to_num, num_to_char, max_len = prepare_data(split_ratio)
            
            # Build and train the model
            model, history, edit_distances = train_model(train_ds, validation_ds, char_to_num, num_to_char, max_len)
            
            # Log additional metrics
            for epoch, edit_dist in enumerate(edit_distances):
                mlflow.log_metric("edit_distance", edit_dist, step=epoch)
            
            # Generate and log plots
            loss_plot_path = create_loss_plot(history)
            edit_distance_plot_path = create_edit_distance_plot(edit_distances)
            
            mlflow.log_artifact(loss_plot_path)
            mlflow.log_artifact(edit_distance_plot_path)
            
            # Create prediction model
            prediction_model = keras.models.Model(
                model.get_layer(name="image").output, model.get_layer(name="dense2").output
            )
            
            # Test the model on a batch and log results
            test_and_log_results(test_ds, prediction_model, num_to_char, max_len)
            
            # Register the model
            mlflow.keras.log_model(
                prediction_model, 
                "prediction_model",
                registered_model_name="HandwritingRecognitionModel"
            )
            
            logger.info(f"Experiment completed successfully. Run ID: {run.info.run_id}")
            
            return run.info.run_id
    
    except Exception as e:
        logger.error(f"Error in experiment: {str(e)}")
        raise

def prepare_data(split_ratio=(0.9, 0.05, 0.05)):
    """
    Prepare the dataset with the specified split ratio
    
    Args:
        split_ratio (tuple): Ratio for train/validation/test split (train, val, test)
        
    Returns:
        tuple: (train_ds, validation_ds, test_ds, char_to_num, num_to_char, max_len)
    """
    try:
        logger.info("Preparing dataset...")
        base_path = "data"
        words_list = []

        words = open(f"{base_path}/words.txt", "r").readlines()
        for line in words:
            if line[0] == "#":
                continue
            if line.split(" ")[1] != "err":  # Skip errored entries
                words_list.append(line)

        logger.info(f"Total samples: {len(words_list)}")
        np.random.shuffle(words_list)

        # Split according to the provided ratio
        train_split_idx = int(split_ratio[0] * len(words_list))
        train_samples = words_list[:train_split_idx]
        remaining_samples = words_list[train_split_idx:]
        
        val_split_idx = int(split_ratio[1] / (split_ratio[1] + split_ratio[2]) * len(remaining_samples))
        validation_samples = remaining_samples[:val_split_idx]
        test_samples = remaining_samples[val_split_idx:]

        logger.info(f"Training samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(validation_samples)}")
        logger.info(f"Testing samples: {len(test_samples)}")

        # Get image paths and labels
        base_image_path = os.path.join(base_path, "words")
        train_img_paths, train_labels = get_image_paths_and_labels(train_samples, base_image_path)
        validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples, base_image_path)
        test_img_paths, test_labels = get_image_paths_and_labels(test_samples, base_image_path)

        # Clean labels
        train_labels_cleaned, characters, max_len = clean_train_labels(train_labels)
        validation_labels_cleaned = clean_labels(validation_labels)
        test_labels_cleaned = clean_labels(test_labels)

        # Build character vocabulary
        char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
        num_to_char = StringLookup(
            vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

        # Prepare datasets
        batch_size = 64
        padding_token = 99
        image_width = 128
        image_height = 32
        
        # Create tf.data.Dataset objects
        train_ds = prepare_dataset(train_img_paths, train_labels_cleaned, char_to_num, max_len, 
                                  batch_size, padding_token, image_width, image_height)
        validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned, char_to_num, max_len,
                                       batch_size, padding_token, image_width, image_height)
        test_ds = prepare_dataset(test_img_paths, test_labels_cleaned, char_to_num, max_len,
                                 batch_size, padding_token, image_width, image_height)
        
        logger.info("Dataset preparation completed")
        return train_ds, validation_ds, test_ds, char_to_num, num_to_char, max_len
    
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise

def get_image_paths_and_labels(samples, base_image_path):
    """Get image paths and corresponding labels"""
    paths = []
    corrected_samples = []
    for file_line in samples:
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.exists(img_path) and os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples

def clean_train_labels(labels):
    """Clean training labels and extract characters and max length"""
    cleaned_labels = []
    characters = set()
    max_len = 0

    for label in labels:
        label = label.split(" ")[-1].strip()
        for char in label:
            characters.add(char)

        max_len = max(max_len, len(label))
        cleaned_labels.append(label)

    characters = sorted(list(characters))
    logger.info(f"Maximum length: {max_len}")
    logger.info(f"Vocabulary size: {len(characters)}")
    
    return cleaned_labels, characters, max_len

def clean_labels(labels):
    """Clean validation and test labels"""
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels

def distortion_free_resize(image, img_size):
    """Resize images without distortion"""
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check the amount of padding needed
    pad_height = h - ops.shape(image)[0]
    pad_width = w - ops.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = ops.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(128, 32)):
    """Preprocess images"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = ops.cast(image, tf.float32) / 255.0
    return image

def vectorize_label(label, char_to_num, max_len, padding_token=99):
    """Convert text labels to sequences of character indices"""
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = ops.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label

def process_images_labels(image_path, label, char_to_num, max_len, padding_token, img_size):
    """Process individual images and labels"""
    image = preprocess_image(image_path, img_size)
    label = vectorize_label(label, char_to_num, max_len, padding_token)
    return {"image": image, "label": label}

def prepare_dataset(image_paths, labels, char_to_num, max_len, batch_size, padding_token, image_width, image_height):
    """Prepare tf.data.Dataset objects"""
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Create a wrapper function with fixed parameters
    def process_wrapper(image_path, label):
        return process_images_labels(
            image_path, label, char_to_num, max_len, padding_token, (image_width, image_height)
        )
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_wrapper, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

class CTCLayer(keras.layers.Layer):
    """CTC Loss Layer"""
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def build_model(char_to_num, image_width=128, image_height=32, max_len=None):
    """Build and compile the model"""
    try:
        logger.info("Building model...")
        # Inputs to the model
        input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
        labels = keras.layers.Input(name="label", shape=(None,))

        # First conv block
        x = keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model.
        new_shape = ((image_width // 4), (image_height // 4) * 64)
        x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
        x = keras.layers.Dropout(0.2)(x)

        # RNNs
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
        )(x)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
        )(x)

        # +2 is to account for the two special tokens introduced by the CTC loss.
        x = keras.layers.Dense(
            len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
        )(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
        )
        
        # Optimizer
        opt = keras.optimizers.Adam()
        
        # Compile the model
        model.compile(optimizer=opt)
        logger.info("Model built successfully")
        
        return model
    
    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        raise

def calculate_edit_distance(labels, predictions, max_len):
    """Calculate the edit distance between predicted and actual labels"""
    # Get a single batch and convert its labels to sparse tensors
    sparse_labels = ops.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.ops.nn.ctc_decode(
        predictions, sequence_lengths=input_len
    )[0][0][:, :max_len]
    sparse_predictions = ops.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out
    edit_distances = tf.edit_distance(
        sparse_predictions, sparse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)

class EditDistanceCallback(keras.callbacks.Callback):
    """Callback to track edit distances during training"""
    def __init__(self, pred_model, validation_ds, max_len):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_images = []
        self.validation_labels = []
        self.max_len = max_len
        self.edit_distances = []
        
        # Extract validation images and labels
        for batch in validation_ds:
            self.validation_images.append(batch["image"])
            self.validation_labels.append(batch["label"])

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(self.validation_images)):
            labels = self.validation_labels[i]
            predictions = self.prediction_model.predict(self.validation_images[i], verbose=0)
            edit_distances.append(calculate_edit_distance(labels, predictions, self.max_len).numpy())

        mean_edit_distance = np.mean(edit_distances)
        self.edit_distances.append(mean_edit_distance)
        logger.info(f"Mean edit distance for epoch {epoch + 1}: {mean_edit_distance:.4f}")

def train_model(train_ds, validation_ds, char_to_num, num_to_char, max_len, epochs=10):
    """Train the model and track metrics"""
    try:
        logger.info("Starting model training...")
        
        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("max_len", max_len)
        mlflow.log_param("vocab_size", len(char_to_num.get_vocabulary()))
        
        # Build model
        model = build_model(char_to_num)
        
        # Create prediction model for edit distance calculation
        prediction_model = keras.models.Model(
            model.get_layer(name="image").output, model.get_layer(name="dense2").output
        )
        
        # Setup the edit distance callback
        edit_distance_callback = EditDistanceCallback(prediction_model, validation_ds, max_len)
        
        # Train the model
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=epochs,
            callbacks=[edit_distance_callback],
        )
        
        logger.info("Model training completed")
        return model, history, edit_distance_callback.edit_distances
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def create_loss_plot(history):
    """Create and save a plot of training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plot_path = 'loss_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def create_edit_distance_plot(edit_distances):
    """Create and save a plot of edit distances over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(edit_distances) + 1), edit_distances, marker='o')
    plt.title('Average Edit Distance per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Edit Distance')
    plt.grid(True)
    
    # Save the plot to a file
    plot_path = 'edit_distance_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def decode_batch_predictions(pred, num_to_char, max_len):
    """Decode model predictions to readable text"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search
    results = keras.ops.nn.ctc_decode(pred, sequence_lengths=input_len)[0][0][:, :max_len]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = (
            tf.strings.reduce_join(num_to_char(res))
            .numpy()
            .decode("utf-8")
            .replace("[UNK]", "")
        )
        output_text.append(res)
    return output_text

def test_and_log_results(test_ds, prediction_model, num_to_char, max_len):
    """Test the model and log sample predictions"""
    logger.info("Testing model on sample images...")
    
    # Create a figure to save
    plt.figure(figsize=(15, 8))
    
    for batch in test_ds.take(1):
        batch_images = batch["image"]
        preds = prediction_model.predict(batch_images, verbose=0)
        pred_texts = decode_batch_predictions(preds, num_to_char, max_len)
        
        # Create a figure with sample results
        _, ax = plt.subplots(4, 4, figsize=(15, 8))
        
        for i in range(min(16, len(batch_images))):
            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = ops.transpose(img, (1, 0, 2))
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]
            
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
            
            # Log individual sample predictions
            mlflow.log_text(f"Sample {i}: {pred_texts[i]}", f"prediction_sample_{i}.txt")
        
        # Save and log the figure
        plt.tight_layout()
        sample_path = "sample_predictions.png"
        plt.savefig(sample_path)
        plt.close()
        mlflow.log_artifact(sample_path)
        
        # Log accuracy metrics for this batch
        correct = 0
        total = 0
        
        if "label" in batch:
            batch_labels = batch["label"]
            for i, label in enumerate(batch_labels):
                if i < len(pred_texts):
                    # Convert label tensor to text
                    label_indices = tf.gather(label, tf.where(tf.math.not_equal(label, 99)))
                    label_text = (
                        tf.strings.reduce_join(num_to_char(label_indices))
                        .numpy()
                        .decode("utf-8")
                    )
                    
                    if pred_texts[i] == label_text:
                        correct += 1
                    total += 1
            
            if total > 0:
                mlflow.log_metric("test_batch_accuracy", correct / total)
        
        logger.info("Model testing completed")

if __name__ == "__main__":
    # Run experiments with different split ratios
    split_ratios = [
        (0.9, 0.05, 0.05),  # Original split
        (0.8, 0.1, 0.1),    # More validation/test data
        (0.85, 0.075, 0.075)  # Intermediate split
    ]
    
    for i, split_ratio in enumerate(split_ratios):
        logger.info(f"Running experiment {i+1} with split ratio: {split_ratio}")
        run_id = main(split_ratio)
        logger.info(f"Experiment {i+1} completed with run_id: {run_id}")