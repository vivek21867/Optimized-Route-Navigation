from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.models import load_model
from PIL import Image
import io
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_FOLDER = 'models'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Disease classes for different datasets
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

# Model paths (these would be populated after training)
MODELS = {
    'resnet152': {
        'ham10000': os.path.join(MODEL_FOLDER, 'resnet152_ham10000.h5'),
        'isic': os.path.join(MODEL_FOLDER, 'resnet152_isic.h5'),
        'fitzpatrick': os.path.join(MODEL_FOLDER, 'resnet152_fitzpatrick.h5')
    },
    'vgg19': {
        'ham10000': os.path.join(MODEL_FOLDER, 'vgg19_ham10000.h5'),
        'isic': os.path.join(MODEL_FOLDER, 'vgg19_isic.h5'),
        'fitzpatrick': os.path.join(MODEL_FOLDER, 'vgg19_fitzpatrick.h5')
    },
    'efficientnet': {
        'ham10000': os.path.join(MODEL_FOLDER, 'efficientnet_ham10000.h5'),
        'isic': os.path.join(MODEL_FOLDER, 'efficientnet_isic.h5'),
        'fitzpatrick': os.path.join(MODEL_FOLDER, 'efficientnet_fitzpatrick.h5')
    }
}

# Health recommendations based on predictions
RECOMMENDATIONS = {
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

# Default recommendations for any condition not specifically listed
DEFAULT_RECOMMENDATIONS = [
    'Consult with a healthcare professional',
    'Keep the area clean and moisturized',
    'Monitor for any changes in appearance'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, model_type, target_size=(224, 224)):
    """Preprocess the image based on the model type"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == 'resnet152':
        return resnet_preprocess(img_array)
    elif model_type == 'vgg19':
        return vgg_preprocess(img_array)
    elif model_type == 'efficientnet':
        return efficientnet_preprocess(img_array)
    else:
        # Default preprocessing
        return img_array / 255.0

def get_model(model_type, dataset):
    """Load the appropriate model based on type and dataset"""
    model_path = MODELS[model_type][dataset]
    
    # Check if model exists
    if not os.path.exists(model_path):
        # In a real application, you might want to train the model if it doesn't exist
        # For this example, we'll just return a message
        return None, f"Model {model_type} for {dataset} not found. Please train the model first."
    
    try:
        model = load_model(model_path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def get_prediction_details(prediction, dataset):
    """Get detailed information about the prediction"""
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index])
    
    if class_index < len(DISEASE_CLASSES[dataset]):
        disease = DISEASE_CLASSES[dataset][class_index]
    else:
        disease = "Unknown"
    
    # Get recommendations for the disease
    if disease in RECOMMENDATIONS:
        recommendations = RECOMMENDATIONS[disease]
    else:
        recommendations = DEFAULT_RECOMMENDATIONS
    
    return {
        'disease': disease,
        'confidence': confidence,
        'recommendations': recommendations
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get model type and dataset from request
    model_type = request.form.get('model_type', 'resnet152')
    dataset = request.form.get('dataset', 'ham10000')
    
    # Validate model type and dataset
    if model_type not in MODELS:
        return jsonify({'error': f'Invalid model type: {model_type}'}), 400
    
    if dataset not in DISEASE_CLASSES:
        return jsonify({'error': f'Invalid dataset: {dataset}'}), 400
    
    try:
        # Save the file temporarily
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)
        
        # Load and preprocess the image
        preprocessed_img = preprocess_image(img_path, model_type)
        
        # Load the model
        model, error = get_model(model_type, dataset)
        if error:
            return jsonify({'error': error}), 500
        
        # Make prediction
        prediction = model.predict(preprocessed_img)
        
        # Get detailed prediction information
        result = get_prediction_details(prediction, dataset)
        
        # Clean up
        os.remove(img_path)
        
        return jsonify({
            'success': True,
            'prediction': result['disease'],
            'confidence': result['confidence'],
            'recommendations': result['recommendations']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    return jsonify({
        'datasets': list(DISEASE_CLASSES.keys()),
        'classes': DISEASE_CLASSES
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': list(MODELS.keys())
    })

if __name__ == '__main__':
    app.run(debug=True)

