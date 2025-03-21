import os
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
DATASET_FOLDER = 'datasets'
PROCESSED_FOLDER = 'processed_datasets'

# Ensure directories exist
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def preprocess_ham10000():
    """Preprocess the HAM10000 dataset"""
    print("Preprocessing HAM10000 dataset...")
    
    # Paths
    ham_path = os.path.join(DATASET_FOLDER, 'ham10000')
    ham_images = os.path.join(ham_path, 'images')
    ham_metadata = os.path.join(ham_path, 'HAM10000_metadata.csv')
    
    # Output paths
    output_path = os.path.join(PROCESSED_FOLDER, 'ham10000')
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Read metadata
    df = pd.read_csv(ham_metadata)
    
    # Create class directories
    for dx in df['dx'].unique():
        os.makedirs(os.path.join(train_path, dx), exist_ok=True)
        os.makedirs(os.path.join(test_path, dx), exist_ok=True)
    
    # Split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])
    
    # Process training images
    print("Processing training images...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img_id = row['image_id']
        dx = row['dx']
        
        # Find image file (could be .jpg or .JPG)
        img_path = None
        for ext in ['.jpg', '.JPG']:
            temp_path = os.path.join(ham_images, img_id + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None:
            print(f"Image {img_id} not found, skipping...")
            continue
        
        # Read and preprocess image
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize to model input size
        
        # Save to train folder
        img.save(os.path.join(train_path, dx, img_id + '.jpg'))
    
    # Process test images
    print("Processing test images...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_id = row['image_id']
        dx = row['dx']
        
        # Find image file
        img_path = None
        for ext in ['.jpg', '.JPG']:
            temp_path = os.path.join(ham_images, img_id + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None:
            print(f"Image {img_id} not found, skipping...")
            continue
        
        # Read and preprocess image
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize to model input size
        
        # Save to test folder
        img.save(os.path.join(test_path, dx, img_id + '.jpg'))
    
    print("HAM10000 preprocessing complete!")

def preprocess_isic():
    """Preprocess the ISIC dataset"""
    print("Preprocessing ISIC dataset...")
    
    # Paths
    isic_path = os.path.join(DATASET_FOLDER, 'isic')
    isic_images = os.path.join(isic_path, 'images')
    isic_metadata = os.path.join(isic_path, 'metadata.csv')
    
    # Output paths
    output_path = os.path.join(PROCESSED_FOLDER, 'isic')
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Read metadata
    df = pd.read_csv(isic_metadata)
    
    # Create class directories
    for diagnosis in df['diagnosis'].unique():
        os.makedirs(os.path.join(train_path, diagnosis), exist_ok=True)
        os.makedirs(os.path.join(test_path, diagnosis), exist_ok=True)
    
    # Split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])
    
    # Process training images
    print("Processing training images...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img_id = row['image_name']
        diagnosis = row['diagnosis']
        
        # Find image file
        img_path = os.path.join(isic_images, img_id + '.jpg')
        
        if not os.path.exists(img_path):
            print(f"Image {img_id} not found, skipping...")
            continue
        
        # Read and preprocess image
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize to model input size
        
        # Save to train folder
        img.save(os.path.join(train_path, diagnosis, img_id + '.jpg'))
    
    # Process test images
    print("Processing test images...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_id = row['image_name']
        diagnosis = row['diagnosis']
        
        # Find image file
        img_path = os.path.join(isic_images, img_id + '.jpg')
        
        if not os.path.exists(img_path):
            print(f"Image {img_id} not found, skipping...")
            continue
        
        # Read and preprocess image
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize to model input size
        
        # Save to test folder
        img.save(os.path.join(test_path, diagnosis, img_id + '.jpg'))
    
    print("ISIC preprocessing complete!")

def preprocess_fitzpatrick():
    """Preprocess the Fitzpatrick 17k dataset"""
    print("Preprocessing Fitzpatrick 17k dataset...")
    
    # Paths
    fitz_path = os.path.join(DATASET_FOLDER, 'fitzpatrick17k')
    fitz_images = os.path.join(fitz_path, 'images')
    fitz_metadata = os.path.join(fitz_path, 'fitzpatrick17k.csv')
    
    # Output paths
    output_path = os.path.join(PROCESSED_FOLDER, 'fitzpatrick')
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Read metadata
    df = pd.read_csv(fitz_metadata)
    
    # Filter to top 9 conditions for simplicity
    top_conditions = df['nine_partition_label'].value_counts().nlargest(9).index.tolist()
    df = df[df['nine_partition_label'].isin(top_conditions)]
    
    # Create class directories
    for condition in top_conditions:
        os.makedirs(os.path.join(train_path, condition), exist_ok=True)
        os.makedirs(os.path.join(test_path, condition), exist_ok=True)
    
    # Split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['nine_partition_label'])
    
    # Process training images
    print("Processing training images...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img_id = row['md5hash']
        condition = row['nine_partition_label']
        
        # Find image file
        img_path = os.path.join(fitz_images, img_id + '.jpg')
        
        if not os.path.exists(img_path):
            print(f"Image {img_id} not found, skipping...")
            continue
        
        # Read and preprocess image
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize to model input size
        
        # Save to train folder
        img.save(os.path.join(train_path, condition, img_id + '.jpg'))
    
    # Process test images
    print("Processing test images...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_id = row['md5hash']
        condition = row['nine_partition_label']
        
        # Find image file
        img_path = os.path.join(fitz_images, img_id + '.jpg')
        
        if not os.path.exists(img_path):
            print(f"Image {img_id} not found, skipping...")
            continue
        
        # Read and preprocess image
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize to model input size
        
        # Save to test folder
        img.save(os.path.join(test_path, condition, img_id + '.jpg'))
    
    print("Fitzpatrick preprocessing complete!")

def visualize_dataset_samples(dataset_name):
    """Visualize sample images from the preprocessed dataset"""
    processed_path = os.path.join(PROCESSED_FOLDER, dataset_name)
    train_path = os.path.join(processed_path, 'train')
    
    # Get all class folders
    class_folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
    
    # Create a figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_folders[:9]):  # Show up to 9 classes
        class_path = os.path.join(train_path, class_name)
        images = os.listdir(class_path)
        
        if images:
            # Get a random image
            img_path = os.path.join(class_path, np.random.choice(images))
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display image
            axes[i].imshow(img)
            axes[i].set_title(class_name)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_FOLDER, f'{dataset_name}_samples.png'))
    plt.close()

def main():
    """Main function to preprocess all datasets"""
    # Preprocess HAM10000
    preprocess_ham10000()
    visualize_dataset_samples('ham10000')
    
    # Preprocess ISIC
    preprocess_isic()
    visualize_dataset_samples('isic')
    
    # Preprocess Fitzpatrick 17k
    preprocess_fitzpatrick()
    visualize_dataset_samples('fitzpatrick')
    
    print("All datasets preprocessed successfully!")

if __name__ == "__main__":
    main()

