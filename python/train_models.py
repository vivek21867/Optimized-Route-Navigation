import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet152, VGG19, EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = (224, 224)
MODEL_FOLDER = 'models'
DATASET_FOLDER = 'datasets'

# Ensure directories exist
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Dataset paths (you would need to download these datasets)
DATASETS = {
    'ham10000': os.path.join(DATASET_FOLDER, 'ham10000'),
    'isic': os.path.join(DATASET_FOLDER, 'isic'),
    'fitzpatrick': os.path.join(DATASET_FOLDER, 'fitzpatrick')
}

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

def create_data_generators(dataset_name):
    """Create data generators for training, validation, and testing"""
    dataset_path = DATASETS[dataset_name]
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation generator
    valid_generator = valid_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Test generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, valid_generator, test_generator

def create_resnet152_model(num_classes):
    """Create a ResNet152 model with custom top layers"""
    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_vgg19_model(num_classes):
    """Create a VGG19 model with custom top layers"""
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_efficientnet_model(num_classes):
    """Create an EfficientNetB4 model with custom top layers"""
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, model_name, dataset_name, train_generator, valid_generator):
    """Train the model and save it"""
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_FOLDER, f'{model_name}_{dataset_name}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

def evaluate_model(model, test_generator, dataset_name, model_name):
    """Evaluate the model and generate performance metrics"""
    # Predict on test data
    test_generator.reset()
    predictions = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = test_generator.classes[:len(predicted_classes)]
    
    # Get class labels
    class_labels = list(test_generator.class_indices.keys())
    
    # Generate classification report
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels,
        output_dict=True
    )
    
    # Convert to DataFrame for easier viewing
    report_df = pd.DataFrame(report).transpose()
    
    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, f'{model_name}_{dataset_name}_cm.png'))
    
    # Save report to CSV
    report_df.to_csv(os.path.join(MODEL_FOLDER, f'{model_name}_{dataset_name}_report.csv'))
    
    return report_df

def fine_tune_model(model, dataset_name, train_generator, valid_generator):
    """Fine-tune the model by unfreezing some layers"""
    # Unfreeze some layers for fine-tuning
    if isinstance(model.layers[0], tf.keras.models.Model):  # Check if using a base model
        base_model = model.layers[0]
        # Unfreeze the last 30 layers
        for layer in base_model.layers[-30:]:
            layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks for fine-tuning
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_FOLDER, f'{model_name}_{dataset_name}_finetuned.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        epochs=20,  # Fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

def main():
    """Main function to train and evaluate models on different datasets"""
    for dataset_name in DATASETS.keys():
        print(f"Processing dataset: {dataset_name}")
        num_classes = len(DISEASE_CLASSES[dataset_name])
        
        # Create data generators
        train_generator, valid_generator, test_generator = create_data_generators(dataset_name)
        
        # Train ResNet152
        print("Training ResNet152 model...")
        resnet_model = create_resnet152_model(num_classes)
        train_model(resnet_model, 'resnet152', dataset_name, train_generator, valid_generator)
        
        # Fine-tune ResNet152
        print("Fine-tuning ResNet152 model...")
        fine_tune_model(resnet_model, dataset_name, train_generator, valid_generator)
        
        # Evaluate ResNet152
        print("Evaluating ResNet152 model...")
        evaluate_model(resnet_model, test_generator, dataset_name, 'resnet152')
        
        # Train VGG19
        print("Training VGG19 model...")
        vgg_model = create_vgg19_model(num_classes)
        train_model(vgg_model, 'vgg19', dataset_name, train_generator, valid_generator)
        
        # Fine-tune VGG19
        print("Fine-tuning VGG19 model...")
        fine_tune_model(vgg_model, dataset_name, train_generator, valid_generator)
        
        # Evaluate VGG19
        print("Evaluating VGG19 model...")
        evaluate_model(vgg_model, test_generator, dataset_name, 'vgg19')
        
        # Train EfficientNet
        print("Training EfficientNet model...")
        efficientnet_model = create_efficientnet_model(num_classes)
        train_model(efficientnet_model, 'efficientnet', dataset_name, train_generator, valid_generator)
        
        # Fine-tune EfficientNet
        print("Fine-tuning EfficientNet model...")
        fine_tune_model(efficientnet_model, dataset_name, train_generator, valid_generator)
        
        # Evaluate EfficientNet
        print("Evaluating EfficientNet model...")
        evaluate_model(efficientnet_model, test_generator, dataset_name, 'efficientnet')

if __name__ == "__main__":
    main()

