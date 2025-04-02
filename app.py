import os
import io
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
from flask import Flask, request, jsonify, render_template, url_for
import logging
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "alzheimer_detection_secret")

# Classification classes
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/model-metrics')
def model_metrics():
    """Render the model metrics page with confusion matrix and evaluation metrics"""
    return render_template('model_metrics.html')

@app.route('/api/confusion-matrix')
def get_confusion_matrix():
    """Generate and return a simulated confusion matrix for visualization"""
    try:
        # Generate a simulated confusion matrix for demonstration
        # In a real application, this would come from model evaluation on a test set
        
        # Define class labels
        class_labels = CLASSES
        num_classes = len(class_labels)
        
        # Simulate a confusion matrix with good performance
        # but some realistic misclassifications
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        # NonDemented (high accuracy)
        confusion_matrix[0, 0] = 95  # True positives
        confusion_matrix[0, 1] = 3   # Misclassified as VeryMildDemented
        confusion_matrix[0, 2] = 1   # Misclassified as MildDemented
        confusion_matrix[0, 3] = 1   # Misclassified as ModerateDemented
        
        # VeryMildDemented (some confusion with NonDemented and MildDemented)
        confusion_matrix[1, 0] = 6   # Misclassified as NonDemented
        confusion_matrix[1, 1] = 86  # True positives
        confusion_matrix[1, 2] = 7   # Misclassified as MildDemented
        confusion_matrix[1, 3] = 1   # Misclassified as ModerateDemented
        
        # MildDemented (some confusion with VeryMildDemented and ModerateDemented)
        confusion_matrix[2, 0] = 2   # Misclassified as NonDemented
        confusion_matrix[2, 1] = 8   # Misclassified as VeryMildDemented
        confusion_matrix[2, 2] = 82  # True positives
        confusion_matrix[2, 3] = 8   # Misclassified as ModerateDemented
        
        # ModerateDemented (some confusion with MildDemented)
        confusion_matrix[3, 0] = 1   # Misclassified as NonDemented
        confusion_matrix[3, 1] = 2   # Misclassified as VeryMildDemented
        confusion_matrix[3, 2] = 7   # Misclassified as MildDemented
        confusion_matrix[3, 3] = 90  # True positives
        
        # Calculate overall accuracy
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix) * 100
        
        # Calculate per-class metrics
        class_metrics = []
        for i in range(num_classes):
            # True positives for this class
            tp = confusion_matrix[i, i]
            
            # False positives (other classes misclassified as this class)
            fp = np.sum(confusion_matrix[:, i]) - tp
            
            # False negatives (this class misclassified as other classes)
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            # True negatives
            tn = np.sum(confusion_matrix) - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics.append({
                'class': class_labels[i],
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1_score * 100, 2),
                'support': int(np.sum(confusion_matrix[i, :]))
            })
        
        # Generate confusion matrix visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64 for sending to frontend
        confusion_matrix_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'confusion_matrix': confusion_matrix.tolist(),
            'class_labels': class_labels,
            'accuracy': round(accuracy, 2),
            'class_metrics': class_metrics,
            'confusion_matrix_img': confusion_matrix_img
        })
        
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_simulated_heatmap(image):
    """
    Create a simulated Grad-CAM heatmap for demonstration purposes
    
    Args:
        image: PIL Image
    
    Returns:
        PIL Image with simulated heatmap overlay
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create a blank heatmap image
    heatmap = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap)
    
    # Generate random "hotspots" for the heatmap - brain regions that might be significant
    # For Alzheimer's, we'll focus on regions like hippocampus, temporal lobe, etc.
    num_hotspots = random.randint(3, 6)  # Increased for more prominent highlighting
    width, height = image.size
    
    # Define area where hotspots are more likely (central brain region)
    center_x_range = (width * 0.3, width * 0.7)
    center_y_range = (height * 0.25, height * 0.75)
    
    for _ in range(num_hotspots):
        # Center coordinates of the hotspot - more focused on central brain regions
        center_x = random.randint(int(center_x_range[0]), int(center_x_range[1]))
        center_y = random.randint(int(center_y_range[0]), int(center_y_range[1]))
        
        # Size of the hotspot - larger for better visibility
        radius = random.randint(width // 8, width // 4)
        
        # Color intensity (red/yellow for hotter areas) - increased for better visibility
        intensity = random.uniform(180, 255)
        
        # Draw a gradient circle for the hotspot
        for r in range(radius, 0, -1):
            # Higher opacity for better visibility
            opacity = int(255 * (1 - r / radius) ** 1.5)  # Steeper falloff for sharper contrast
            
            if r > radius * 0.6:  # Adjusted threshold for wider yellow region
                # Outer region: more yellow/orange with higher saturation
                color = (255, int(intensity * 0.9), 0, opacity)
            else:
                # Inner region: more red with higher saturation
                color = (255, int(intensity * 0.2), 0, opacity)
            
            draw.ellipse(
                [(center_x - r, center_y - r), (center_x + r, center_y + r)],
                fill=color
            )
    
    # Apply a slight blur to smooth the heatmap, reduced for sharper edges
    heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Overlay the heatmap on the original image
    # Convert original image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Darken the original image more to increase contrast with the heatmap
    darkened = ImageEnhance.Brightness(image).enhance(0.6)
    
    # Increase contrast to make brain structures more visible
    darkened = ImageEnhance.Contrast(darkened).enhance(1.2)
    
    # Convert to RGBA for compositing
    darkened_rgba = darkened.convert('RGBA')
    
    # Overlay the heatmap with higher alpha blend
    result = Image.alpha_composite(darkened_rgba, heatmap)
    
    # Add a subtle border or glow around the hottest areas for better distinction
    # This is achieved by dilating the brightest parts slightly
    heatmap_enhanced = result.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    return heatmap_enhanced.convert('RGB')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to process the image and return predictions"""
    try:
        # Get the file from the FormData
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        # Ensure it's an image
        if not file.content_type.startswith('image/'):
            logger.error(f"Invalid file type: {file.content_type}")
            return jsonify({'error': 'Please upload a valid image file'}), 400
        
        # Read the image safely
        try:
            # Read image into memory first to avoid potential streaming issues
            image_bytes = file.read()
            if not image_bytes:
                logger.error("Empty file uploaded")
                return jsonify({'error': 'Empty file uploaded'}), 400
                
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize the image to a reasonable size
            image.thumbnail((400, 400))
            
            # For demonstration, generate a random prediction
            # In a real system, this would use a trained model
            class_index = random.randint(0, 3)
            confidence = random.uniform(0.7, 0.98)
            predicted_class = CLASSES[class_index]
            
            # Generate a simulated Grad-CAM visualization
            cam_image = create_simulated_heatmap(image)
            
            # Convert the CAM image to base64 for sending to frontend
            buffered = io.BytesIO()
            cam_image.save(buffered, format="JPEG")
            cam_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Also send the original image
            orig_buffered = io.BytesIO()
            image.save(orig_buffered, format="JPEG")
            orig_image_base64 = base64.b64encode(orig_buffered.getvalue()).decode('utf-8')
            
            return jsonify({
                'class': predicted_class,
                'confidence': confidence,
                'class_index': class_index,
                'cam_image': cam_image_base64,
                'original_image': orig_image_base64
            })
            
        except IOError as io_err:
            logger.error(f"IOError when reading image: {str(io_err)}")
            return jsonify({'error': 'Invalid or corrupted image file'}), 400
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
