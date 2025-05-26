import tensorflow as tf
from img_clas import create_solar_panel_defect_detector, get_class_info, yolo_loss  # Import the custom loss
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches
import random  # Add this import at the top

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get class info
_, class_names, _ = get_class_info("Solar Panel Fault Dataset.v1i.coco/train/_annotations.coco.json")
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Check what files exist
print("Checking for model files...")
if os.path.exists('solar_panel_defect_model'):
    for root, dirs, files in os.walk('solar_panel_defect_model'):
        for file in files:
            print(f"  Found: {os.path.join(root, file)}")

# Robust model loading with custom loss function
model = None

def load_weights_method():
    """Load weights into a new model instance"""
    model = create_solar_panel_defect_detector(num_classes)
    # Try different weight paths
    weight_paths = [
        'solar_panel_defect_model/model_checkpoint',
        'solar_panel_defect_model/final_model/variables/variables',
        'solar_panel_defect_model/checkpoint'
    ]
    
    for weight_path in weight_paths:
        try:
            model.load_weights(weight_path)
            print(f"✓ Weights loaded from: {weight_path}")
            return model
        except Exception as e:
            print(f"✗ Failed to load from {weight_path}: {e}")
            continue
    
    raise Exception("No valid weight files found")

# Try multiple loading methods in order
loading_methods = [
    # Method 1: Load full model with custom loss
    lambda: tf.keras.models.load_model(
        'solar_panel_defect_model/final_model',
        custom_objects={'yolo_loss': yolo_loss}
    ),
    # Method 2: Load without compiling (ignore loss function)
    lambda: tf.keras.models.load_model(
        'solar_panel_defect_model/final_model',
        compile=False
    ),
    # Method 3: Load weights only
    load_weights_method
]

for i, method in enumerate(loading_methods, 1):
    try:
        model = method()
        print(f"✓ Model loaded successfully using method {i}")
        break
    except Exception as e:
        print(f"✗ Method {i} failed: {e}")

if model is None:
    print("ERROR: Could not load any model.")
    print("Available files:")
    if os.path.exists('solar_panel_defect_model'):
        for root, dirs, files in os.walk('solar_panel_defect_model'):
            for file in files:
                print(f"  {os.path.join(root, file)}")
    exit(1)

print("✓ Model ready for testing!")

# Non-max suppression function
def non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5):
    selected_indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=100,
        iou_threshold=iou_threshold
    )
    
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)
    selected_classes = tf.gather(class_ids, selected_indices)
    
    return selected_boxes, selected_scores, selected_classes

# Test function for single image
def test_single_image(model, image_path, class_names):
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    original_img = img  # Keep original for display
    img_resized = tf.image.resize(img, (224, 224))
    img_normalized = tf.cast(img_resized, tf.float32) / 255.0
    img_batch = tf.expand_dims(img_normalized, 0)
    
    # Make prediction
    prediction = model.predict(img_batch)[0]
    
    # Process prediction grid
    grid_size = 7
    cell_size = 224 / grid_size
    
    # Lists for detections
    detected_boxes = []
    detected_scores = []
    detected_classes = []
    
    confidence_threshold = 0.3
    
    for row in range(grid_size):
        for col in range(grid_size):
            cell_pred = prediction[row, col]
            confidence = cell_pred[0]
            
            if confidence > confidence_threshold:
                # Extract coordinates
                cx = (col + cell_pred[1]) * cell_size
                cy = (row + cell_pred[2]) * cell_size
                w = cell_pred[3] * 224
                h = cell_pred[4] * 224
                
                # Convert to box coordinates
                x_min = cx - w/2
                y_min = cy - h/2
                x_max = x_min + w
                y_max = y_min + h
                
                # Get class
                class_probs = cell_pred[5:]
                class_id = tf.argmax(class_probs)
                
                detected_boxes.append([x_min, y_min, x_max, y_max])
                detected_scores.append(confidence)
                detected_classes.append(class_id)
    
    # Apply NMS
    if detected_boxes:
        boxes_tensor = tf.convert_to_tensor(detected_boxes, dtype=tf.float32)
        scores_tensor = tf.convert_to_tensor(detected_scores, dtype=tf.float32)
        classes_tensor = tf.convert_to_tensor(detected_classes, dtype=tf.int32)
        
        boxes_nms, scores_nms, classes_nms = non_max_suppression(
            boxes_tensor, scores_tensor, classes_tensor
        )
        
        # Visualize result
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(tf.cast(original_img, tf.uint8))
        plt.title('Original Image')
        plt.axis('off')
        
        # Predictions
        plt.subplot(1, 2, 2)
        plt.imshow(img_normalized)
        plt.title('Detected Defects')
        plt.axis('off')
        
        for box, score, class_id in zip(boxes_nms.numpy(), scores_nms.numpy(), classes_nms.numpy()):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height,
                                   linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x_min, y_min - 5, 
                    f"{class_names[class_id]} ({score:.2f})",
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
        
        plt.tight_layout()
        plt.show()
        
        print(f"Found {len(boxes_nms)} defects:")
        for i, (box, score, class_id) in enumerate(zip(boxes_nms.numpy(), scores_nms.numpy(), classes_nms.numpy())):
            print(f"  {i+1}. {class_names[class_id]} - Confidence: {score:.3f}")
        
        return boxes_nms, scores_nms, classes_nms
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(tf.cast(original_img, tf.uint8))
        plt.title('No Defects Detected')
        plt.axis('off')
        plt.show()
        print("No defects detected in the image.")
        return None, None, None

# Compact grid display for multiple test images
def test_images_compact_grid(model, test_dir, class_names, num_images=10):
    """Show multiple random images in a compact grid"""
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly select images
    if len(test_images) > num_images:
        test_images = random.sample(test_images, num_images)
    else:
        num_images = len(test_images)
    
    print(f"Testing {num_images} random images from {len(test_images)} available")
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(num_images):
        test_image_path = os.path.join(test_dir, test_images[i])
        print(f"Processing random image {i+1}: {test_images[i]}")
        
        # Load and predict
        img = tf.io.read_file(test_image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img_resized = tf.image.resize(img, (224, 224))
        img_normalized = tf.cast(img_resized, tf.float32) / 255.0
        img_batch = tf.expand_dims(img_normalized, 0)
        prediction = model.predict(img_batch, verbose=0)[0]
        
        # Process predictions (simplified)
        detected_count = 0
        grid_size = 7
        cell_size = 224 / grid_size
        confidence_threshold = 0.3
        
        ax = axes[i]
        ax.imshow(img_normalized)
        
        # Quick detection count and drawing
        for row in range(grid_size):
            for col in range(grid_size):
                cell_pred = prediction[row, col]
                confidence = cell_pred[0]
                
                if confidence > confidence_threshold:
                    detected_count += 1
                    cx = (col + cell_pred[1]) * cell_size
                    cy = (row + cell_pred[2]) * cell_size
                    w = cell_pred[3] * 224
                    h = cell_pred[4] * 224
                    
                    x_min = cx - w/2
                    y_min = cy - h/2
                    width = w
                    height = h
                    
                    rect = patches.Rectangle((x_min, y_min), width, height,
                                           linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
        
        ax.set_title(f'{test_images[i][:15]}... ({detected_count} defects)', fontsize=8)
        ax.axis('off')
        
        print(f"  - {detected_count} defects detected")
    
    # Hide unused subplots
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Updated compact grid display for multiple test images with defect types
def test_images_compact_grid(model, test_dir, class_names, num_images=10):
    """Show multiple random images in a compact grid with defect types"""
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly select images
    if len(test_images) > num_images:
        test_images = random.sample(test_images, num_images)
    else:
        num_images = len(test_images)
    
    print(f"Testing {num_images} random images from {len(test_images)} available")
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))  # Increased width for better text visibility
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(num_images):
        test_image_path = os.path.join(test_dir, test_images[i])
        print(f"Processing random image {i+1}: {test_images[i]}")
        
        # Load and predict
        img = tf.io.read_file(test_image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img_resized = tf.image.resize(img, (224, 224))
        img_normalized = tf.cast(img_resized, tf.float32) / 255.0
        img_batch = tf.expand_dims(img_normalized, 0)
        prediction = model.predict(img_batch, verbose=0)[0]
        
        # Process predictions with NMS for better accuracy
        grid_size = 7
        cell_size = 224 / grid_size
        confidence_threshold = 0.3
        
        detected_boxes = []
        detected_scores = []
        detected_classes = []
        
        ax = axes[i]
        ax.imshow(img_normalized)
        
        # Extract all detections first
        for row in range(grid_size):
            for col in range(grid_size):
                cell_pred = prediction[row, col]
                confidence = cell_pred[0]
                
                if confidence > confidence_threshold:
                    cx = (col + cell_pred[1]) * cell_size
                    cy = (row + cell_pred[2]) * cell_size
                    w = cell_pred[3] * 224
                    h = cell_pred[4] * 224
                    
                    x_min = cx - w/2
                    y_min = cy - h/2
                    x_max = x_min + w
                    y_max = y_min + h
                    
                    # Get class prediction
                    class_probs = cell_pred[5:]
                    class_id = tf.argmax(class_probs)
                    
                    detected_boxes.append([x_min, y_min, x_max, y_max])
                    detected_scores.append(confidence)
                    detected_classes.append(class_id)
        
        # Apply NMS and draw boxes with class names
        if detected_boxes:
            boxes_tensor = tf.convert_to_tensor(detected_boxes, dtype=tf.float32)
            scores_tensor = tf.convert_to_tensor(detected_scores, dtype=tf.float32)
            classes_tensor = tf.convert_to_tensor(detected_classes, dtype=tf.int32)
            
            boxes_nms, scores_nms, classes_nms = non_max_suppression(
                boxes_tensor, scores_tensor, classes_tensor
            )
            
            # Draw bounding boxes with defect types
            detected_defects = []
            for box, score, class_id in zip(boxes_nms.numpy(), scores_nms.numpy(), classes_nms.numpy()):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                
                # Get defect type name
                defect_type = class_names[class_id]
                detected_defects.append(defect_type)
                
                # Draw bounding box
                rect = patches.Rectangle((x_min, y_min), width, height,
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add text with defect type and confidence
                ax.text(x_min, y_min - 2, 
                       f"{defect_type}\n{score:.2f}",  # Defect type on first line, confidence on second
                       fontsize=8, color='white', weight='bold',
                       bbox=dict(facecolor='red', alpha=0.8, boxstyle='round,pad=0.2'))
            
            # Create title with defect types
            if detected_defects:
                unique_defects = list(set(detected_defects))  # Remove duplicates
                defects_text = ", ".join(unique_defects[:2])  # Show max 2 defect types
                if len(unique_defects) > 2:
                    defects_text += f" +{len(unique_defects)-2} more"
                title = f'{test_images[i][:12]}...\n{len(detected_defects)} defects: {defects_text}'
            else:
                title = f'{test_images[i][:15]}...\nNo defects'
            
            print(f"  - {len(detected_defects)} defects: {detected_defects}")
        else:
            title = f'{test_images[i][:15]}...\nNo defects'
            print(f"  - No defects detected")
        
        ax.set_title(title, fontsize=9, pad=10)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return  # Optional: return summary statistics

# Enhanced version with detailed summary
def test_images_with_defect_summary(model, test_dir, class_names, num_images=10):
    """Test images and provide detailed defect type summary"""
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly select images
    if len(test_images) > num_images:
        test_images = random.sample(test_images, num_images)
    else:
        num_images = len(test_images)
    
    print(f"Testing {num_images} random images from {len(test_images)} available")
    
    # Track defect statistics
    defect_counts = {class_name: 0 for class_name in class_names}
    total_defects = 0
    images_with_defects = 0
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(num_images):
        test_image_path = os.path.join(test_dir, test_images[i])
        
        # Load and predict
        img = tf.io.read_file(test_image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img_resized = tf.image.resize(img, (224, 224))
        img_normalized = tf.cast(img_resized, tf.float32) / 255.0
        img_batch = tf.expand_dims(img_normalized, 0)
        prediction = model.predict(img_batch, verbose=0)[0]
        
        # Process predictions
        grid_size = 7
        cell_size = 224 / grid_size
        confidence_threshold = 0.3
        
        detected_boxes = []
        detected_scores = []
        detected_classes = []
        
        for row in range(grid_size):
            for col in range(grid_size):
                cell_pred = prediction[row, col]
                confidence = cell_pred[0]
                
                if confidence > confidence_threshold:
                    cx = (col + cell_pred[1]) * cell_size
                    cy = (row + cell_pred[2]) * cell_size
                    w = cell_pred[3] * 224
                    h = cell_pred[4] * 224
                    
                    x_min = cx - w/2
                    y_min = cy - h/2
                    x_max = x_min + w
                    y_max = y_min + h
                    
                    class_probs = cell_pred[5:]
                    class_id = tf.argmax(class_probs)
                    
                    detected_boxes.append([x_min, y_min, x_max, y_max])
                    detected_scores.append(confidence)
                    detected_classes.append(class_id)
        
        ax = axes[i]
        ax.imshow(img_normalized)
        
        image_defects = []
        if detected_boxes:
            boxes_tensor = tf.convert_to_tensor(detected_boxes, dtype=tf.float32)
            scores_tensor = tf.convert_to_tensor(detected_scores, dtype=tf.float32)
            classes_tensor = tf.convert_to_tensor(detected_classes, dtype=tf.int32)
            
            boxes_nms, scores_nms, classes_nms = non_max_suppression(
                boxes_tensor, scores_tensor, classes_tensor
            )
            
            for box, score, class_id in zip(boxes_nms.numpy(), scores_nms.numpy(), classes_nms.numpy()):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                
                defect_type = class_names[class_id]
                image_defects.append(defect_type)
                defect_counts[defect_type] += 1
                
                # Draw box with defect type
                rect = patches.Rectangle((x_min, y_min), width, height,
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                ax.text(x_min, y_min - 2, 
                       f"{defect_type}\n{score:.2f}",
                       fontsize=8, color='white', weight='bold',
                       bbox=dict(facecolor='red', alpha=0.8, boxstyle='round,pad=0.2'))
            
            if image_defects:
                images_with_defects += 1
                total_defects += len(image_defects)
        
        # Set title
        if image_defects:
            unique_defects = list(set(image_defects))
            defects_text = ", ".join(unique_defects[:2])
            if len(unique_defects) > 2:
                defects_text += f" +{len(unique_defects)-2}"
            title = f'{test_images[i][:12]}...\n{len(image_defects)}: {defects_text}'
        else:
            title = f'{test_images[i][:15]}...\nNo defects'
        
        ax.set_title(title, fontsize=9, pad=10)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("DEFECT DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Images tested: {num_images}")
    print(f"Images with defects: {images_with_defects}")
    print(f"Detection rate: {images_with_defects/num_images*100:.1f}%")
    print(f"Total defects found: {total_defects}")
    print(f"Average defects per image: {total_defects/num_images:.2f}")
    print(f"\nDefect types found:")
    for defect_type, count in defect_counts.items():
        if count > 0:
            print(f"  {defect_type}: {count} instances")

# Usage:
if __name__ == "__main__":
    # Test on a single image
    test_image_path = "Solar Panel Fault Dataset.v1i.coco/test/image_name.jpg"  # Replace with actual image path
    
    # Or list all test images and pick one
    test_dir = "Solar Panel Fault Dataset.v1i.coco/test"
    if os.path.exists(test_dir):
        test_images_compact_grid(model, test_dir, class_names, num_images=10)
    else:
        print("Test directory not found!")