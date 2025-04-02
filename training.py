import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from model import create_cnn_lstm_model

def train_model(dataset_path, model_save_path='alzheimer_cnn_lstm_model.h5'):
    """
    Train the CNN-LSTM model on the Alzheimer's MRI dataset
    
    Args:
        dataset_path: Path to the dataset directory
        model_save_path: Path to save the trained model
    """
    # Constants
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Generate training data
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Generate validation data
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    print(f"Classes: {class_names}")
    
    # Create model
    model = create_cnn_lstm_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=len(class_names)
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
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
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Evaluate the model
    validation_generator.reset()
    Y_pred = model.predict(validation_generator, validation_generator.samples // BATCH_SIZE + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    # Get true labels
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    
    # Print classification report
    print(classification_report(true_classes, y_pred, target_names=class_labels))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_classes, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model, history

if __name__ == "__main__":
    # Example usage:
    # Assume the dataset is organized in a directory with subdirectories for each class
    # For example:
    # dataset/
    #   NonDemented/
    #       image1.jpg
    #       image2.jpg
    #       ...
    #   VeryMildDemented/
    #       image1.jpg
    #       ...
    #   MildDemented/
    #       ...
    #   ModerateDemented/
    #       ...
    
    # The dataset path should point to the parent directory containing these class subdirectories
    # You would need to download the dataset from lukechugh/best-alzheimer-mri-dataset-99-accuracy
    # and organize it in this structure
    dataset_path = "dataset"  # Replace with actual path
    
    if os.path.exists(dataset_path):
        model, history = train_model(dataset_path)
        print("Model training completed successfully!")
    else:
        print(f"Dataset not found at {dataset_path}. Please download and organize the dataset first.")
        print("You can find the dataset at lukechugh/best-alzheimer-mri-dataset-99-accuracy")
