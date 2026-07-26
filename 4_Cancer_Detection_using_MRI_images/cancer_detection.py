import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import medmnist
from medmnist import INFO, Evaluator

print("Loading BreastMNIST dataset for cancer detection...")
data_flag = 'breastmnist'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# Load the data
train_dataset = DataClass(split='train', download=True)
test_dataset = DataClass(split='test', download=True)

# Prepare data for Keras
# medmnist dataset gives PIL images or numpy arrays, we convert to numpy
X_train = np.array([np.array(img) for img, _ in train_dataset])
y_train = np.array([label for _, label in train_dataset])
X_test = np.array([np.array(img) for img, _ in test_dataset])
y_test = np.array([label for _, label in test_dataset])

print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Preprocess: reshape to add channel dimension and normalize
X_train = X_train.reshape((-1, 28, 28, n_channels)) / 255.0
X_test = X_test.reshape((-1, 28, 28, n_channels)) / 255.0

print("Building CNN model...")
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, n_channels)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot a few images and their predictions
y_pred_logits = model.predict(X_test)
y_pred = np.argmax(y_pred_logits, axis=1)

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # The image is 28x28x1, drop the last channel to plot
    plt.imshow(X_test[i, ..., 0], cmap='gray')
    pred_class = info['label'][str(y_pred[i])]
    true_class = info['label'][str(y_test[i][0])]
    plt.title(f"P:{pred_class}\nT:{true_class}", fontsize=8)
    plt.axis('off')
    
plt.tight_layout()
plt.savefig('cancer_detection_results.png')
print("Saved cancer_detection_results.png")
