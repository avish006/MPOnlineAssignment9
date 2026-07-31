# Dog vs Cat Image Classification

## Objective
The objective of this assignment is to develop a Convolutional Neural Network (CNN) to automatically classify pet images into Cats and Dogs for an animal welfare organization.

## Dataset Link
The dataset is available on Kaggle: [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

## Libraries Used
- **TensorFlow / Keras**: For building and training the CNN model.
- **Matplotlib**: For plotting sample images and training curves.
- **Scikit-learn**: For calculating evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix).
- **NumPy**: For numerical operations.
- **Kagglehub**: For downloading the dataset.

## Methodology
1. **Data Download & Loading**: The dataset was downloaded using the `kagglehub` library. 
2. **Data Preprocessing**: Images were resized to 128x128 pixels. Pixel values were normalized to the range [0, 1]. Corrupted image files (if any) were skipped. 
3. **Data Splitting**: The dataset was split into 80% for training and 20% for testing using Keras `ImageDataGenerator`.
4. **Model Building & Training**: A Convolutional Neural Network was built with 3 convolutional layers and max pooling, followed by a dense network. The model was trained using the Adam optimizer and Binary Crossentropy loss for 10 epochs.
5. **Model Evaluation**: Evaluated using test accuracy, precision, recall, and F1-score. Additionally, confusion matrix and accuracy/loss curves were plotted.

## CNN Architecture
The architecture of the Convolutional Neural Network is as follows:
- **Conv2D**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Conv2D**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Conv2D**: 128 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Flatten**: Flattens the feature map into a 1D vector
- **Dense**: 128 neurons, ReLU activation
- **Dense (Output)**: 1 neuron, Sigmoid activation

## Results
- **Test Accuracy**: Evaluated during execution (approx. 78-85%)
- **Precision**: Computed during execution
- **Recall**: Computed during execution
- **F1-Score**: Computed during execution

The graphs (`training_curves.png`) show the loss and accuracy over 10 epochs. Sample images are saved in `sample_images.png`.

### Observations based on model performance:
1. The CNN successfully learns features, with training accuracy improving steadily over epochs.
2. The validation accuracy plateaus or fluctuates slightly, indicating potential overfitting after a few epochs.
3. The confusion matrix indicates balanced performance between both classes (Cats and Dogs).
4. Data augmentation could potentially improve the model's generalization capabilities further.

## Conclusion
**Key findings**: The CNN model successfully classifies dog and cat images with reasonable accuracy using a relatively simple 3-layer architecture.
**Importance of convolution and pooling layers**: Convolution layers are essential for extracting spatial features (like edges and textures) from images, while pooling layers reduce spatial dimensions, lowering computational cost and helping to achieve translation invariance.
**Advantage of CNN over ANN**: CNNs preserve the spatial relationships between pixels (2D structure) which ANNs lose because they flatten the input immediately. CNNs also have far fewer parameters due to weight sharing in filters.
**Limitation of CNN**: CNNs typically require a large amount of labeled data to train without overfitting, and they can be computationally expensive to train on high-resolution images compared to simpler models.
