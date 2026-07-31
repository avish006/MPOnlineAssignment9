import os
import kagglehub
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def main():
    # 1. Download and load the dataset
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download('bhavikjikadara/dog-and-cat-classification-dataset')
    print(f"Dataset path: {dataset_path}")

    pet_images_path = os.path.join(dataset_path, 'PetImages')
    cat_dir = os.path.join(pet_images_path, 'Cat')
    dog_dir = os.path.join(pet_images_path, 'Dog')

    # Remove corrupted files like Thumbs.db if they exist
    for root, _, files in os.walk(pet_images_path):
        for file in files:
            if file.endswith('.db'):
                os.remove(os.path.join(root, file))

    # 2. Display folder structure
    print("\nFolder Structure:")
    print(f"- {dataset_path}")
    print(f"  - PetImages")
    print(f"    - Cat ({len(os.listdir(cat_dir))} images)")
    print(f"    - Dog ({len(os.listdir(dog_dir))} images)")

    # 3. Display five sample images with class labels
    sample_images = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)[:3]] + \
                    [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)[:2]]
    sample_labels = ['Cat'] * 3 + ['Dog'] * 2

    plt.figure(figsize=(15, 5))
    for i in range(5):
        img = tf.keras.preprocessing.image.load_img(sample_images[i], target_size=(128, 128))
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(sample_labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("\nSaved 5 sample images to 'sample_images.png'.")

    # 4. Identify
    num_classes = 2
    img_dim = "128x128" # from preprocessing
    total_images = len(os.listdir(cat_dir)) + len(os.listdir(dog_dir))
    print(f"\nNumber of classes: {num_classes}")
    print(f"Image dimensions (resized): {img_dim}")
    print(f"Total number of images: {total_images}")

    # Task 2: Data Preprocessing
    print("\nPreprocessing data...")
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        pet_images_path,
        target_size=(128, 128),
        batch_size=64,
        class_mode='binary',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        pet_images_path,
        target_size=(128, 128),
        batch_size=64,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Task 3: Model Development
    print("\nBuilding model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nTraining model...")
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    # Task 4: Model Evaluation
    print("\nEvaluating model...")
    val_generator.reset()
    Y_pred = model.predict(val_generator)
    y_pred = np.where(Y_pred > 0.5, 1, 0)
    y_true = val_generator.classes

    test_accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # Plotting Accuracy and Loss
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("\nSaved accuracy and loss graphs to 'training_curves.png'.")

if __name__ == "__main__":
    main()
