import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess an image for the CNN-LSTM model
    
    Args:
        image: PIL Image
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image as numpy array
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to target size
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    return img_array

def grad_cam(model, img_array, class_idx, layer_name=None):
    """
    Generate Grad-CAM visualization for the given image and class
    
    Args:
        model: Trained model
        img_array: Preprocessed image (single image, not batched)
        class_idx: Index of the target class
        layer_name: Name of the layer to use for Grad-CAM (if None, uses the last conv layer)
    
    Returns:
        PIL Image with Grad-CAM overlay, raw heatmap
    """
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    # Check if we found a convolutional layer
    if layer_name is None:
        # If no convolutional layer found, just return the original image
        original_img = Image.fromarray((img_array * 255).astype(np.uint8))
        return original_img, None
    
    # Create a model that outputs both the predictions and the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Add a batch dimension to the input image
    img_array_batched = np.expand_dims(img_array, axis=0)
    
    # Compute the gradient of the top predicted class for the input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array_batched)
        loss = predictions[:, class_idx]
    
    # Extract the gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map by the gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize the heatmap to the size of the input image
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # Convert the heatmap to RGB
    heatmap_rgb = np.uint8(255 * heatmap_resized)
    heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
    
    # Convert the original image to BGR (for OpenCV)
    img_bgr = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Superimpose the heatmap on the image
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_rgb, 0.4, 0)
    
    # Convert back to RGB for PIL
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    # Create PIL image
    result_img = Image.fromarray(superimposed_img_rgb)
    
    return result_img, heatmap_resized
