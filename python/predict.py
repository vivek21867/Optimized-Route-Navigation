import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
MODEL_FOLDER = 'models'
DISEASE_CLASSES = {
    'ham10000': [
        'Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
        'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions'
    ],
    'isic': [
        'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma',
        'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 
        'Vascular lesion', 'Squamous cell carcinoma'
    ],
    'fitzpatrick': [
        'Acne', 'Eczema', 'Melanoma', 'Psoriasis', 'Vitiligo',
        'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Rosacea', 'Tinea'
    ]
}

def preprocess_image(img_path, model_type):
    """Preprocess the image based on the model type"""
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Ensure we have 3 channels
    if img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize based on model type
    if model_type == 'resnet152':
        # ResNet preprocessing
        img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    elif model_type == 'vgg19':
        # VGG19 preprocessing
        img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    elif model_type == 'efficientnet':
        # EfficientNet preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    else:
        # Default preprocessing
        img_array = img_array / 255.0
    
    return img_array

def predict_image(img_path, model_type='resnet152', dataset='ham10000'):
    """Predict the skin disease from an image"""
    # Check if model exists
    model_path = os.path.join(MODEL_FOLDER, f'{model_type}_{dataset}.h5')
    if not os.path.exists(model_path):
        return {
            'error': f"Model {model_type} for {dataset} not found. Please train the model first."
        }
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Preprocess image
        preprocessed_img = preprocess_image(img_path, model_type)
        
        # Make prediction
        prediction = model.predict(preprocessed_img)
        
        # Get class with highest probability
        class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][class_index])
        
        # Get class name
        if class_index < len(DISEASE_CLASSES[dataset]):
            disease = DISEASE_CLASSES[dataset][class_index]
        else:
            disease = "Unknown"
        
        # Get recommendations
        recommendations = get_recommendations(disease)
        
        # Create heatmap visualization
        create_heatmap(img_path, model, model_type, class_index)
        
        return {
            'disease': disease,
            'confidence': confidence,
            'recommendations': recommendations,
            'heatmap': 'heatmap.jpg'  # Path to generated heatmap
        }
    
    except Exception as e:
        return {'error': str(e)}

def get_recommendations(disease):
    """Get recommendations based on the predicted disease"""
    recommendations = {
        'Melanoma': [
            'Consult a dermatologist immediately',
            'Avoid sun exposure to the affected area',
            'Document any changes in the lesion'
        ],
        'Basal cell carcinoma': [
            'Seek medical attention promptly',
            'Protect the area from sun exposure',
            'Avoid scratching or irritating the lesion'
        ],
        'Actinic keratoses': [
            'Schedule a dermatology appointment',
            'Use sunscreen regularly',
            'Monitor for changes in size or appearance'
        ],
        'Acne': [
            'Wash affected areas twice daily',
            'Avoid touching or picking at the area',
            'Consider over-the-counter acne products with benzoyl peroxide'
        ],
        'Eczema': [
            'Keep the skin moisturized',
            'Avoid known triggers',
            'Consider over-the-counter hydrocortisone cream'
        ],
        'Psoriasis': [
            'Moisturize regularly',
            'Avoid triggers like stress',
            'Consider light therapy or prescribed medications'
        ]
    }
    
    # Return recommendations for the disease or default recommendations
    return recommendations.get(disease, [
        'Consult with a healthcare professional',
        'Keep the area clean and moisturized',
        'Monitor for any changes in appearance'
    ])

def create_heatmap(img_path, model, model_type, class_index):
    """Create a heatmap visualization using Grad-CAM"""
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Normalize based on model type
        if model_type == 'resnet152':
            x = tf.keras.applications.resnet.preprocess_input(x)
        elif model_type == 'vgg19':
            x = tf.keras.applications.vgg19.preprocess_input(x)
        elif model_type == 'efficientnet':
            x = tf.keras.applications.efficientnet.preprocess_input(x)
        else:
            x = x / 255.0
        
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            print("Could not find convolutional layer for heatmap")
            return
        
        # Create Grad-CAM model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        
        # Compute gradient of the predicted class with respect to the output feature map
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            loss = predictions[:, class_index]
        
        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image size
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        # Save heatmap
        cv2.imwrite('heatmap.jpg', superimposed_img)
        
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Predict skin disease from image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default='resnet152', choices=['resnet152', 'vgg19', 'efficientnet'], help='Model type')
    parser.add_argument('--dataset', type=str, default='ham10000', choices=['ham10000', 'isic', 'fitzpatrick'], help='Dataset')
    
    args = parser.parse_args()
    
    result = predict_image(args.image, args.model, args.dataset)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Predicted disease: {result['disease']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"- {rec}")
        
        if 'heatmap' in result:
            print(f"\nHeatmap visualization saved to {result['heatmap']}")

if __name__ == "__main__":
    main()

