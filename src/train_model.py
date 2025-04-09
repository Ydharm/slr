import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)

def create_model(num_classes):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten layer
        Flatten(),
        
        # Dense layers
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Dataset path
    data_dir = 'data/dataset'  # Make sure your dataset is in this directory
    
    # Create data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Parameters
    img_size = 64
    batch_size = 32

    # Create training generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Create validation generator
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print("Class indices:", train_generator.class_indices)

    # Create model
    model = create_model(num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    print("\nEvaluating model on validation data...")
    evaluation = model.evaluate(validation_generator)
    print(f"Validation Loss: {evaluation[0]:.4f}")
    print(f"Validation Accuracy: {evaluation[1]:.4f}")

    # Save the final model
    model.save('models/final_model.h5')
    print("\nModel saved successfully!")

    # Save class indices
    import json
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists('data/dataset'):
        print("Error: Dataset not found!")
        print("Please create a 'data/dataset' directory with your sign language images")
        print("The directory structure should be:")
        print("data/dataset/")
        print("  ├── A/")
        print("  ├── B/")
        print("  ├── C/")
        print("  └── ... (one directory for each sign)")
        exit(1)
    
    main()