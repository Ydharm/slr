import os
import tensorflow as tf
from .model import create_model
from .preprocess import create_data_generators

def train_model():
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    
    # Create data generators
    train_generator, validation_generator = create_data_generators(data_dir)
    
    # Get number of classes
    num_classes = len(os.listdir(data_dir))
    
    # Create and train model
    model = create_model(num_classes)
    
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # Save the model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'sign_language_model.h5')
    model.save(model_path)
    
    return history

if __name__ == "__main__":
    history = train_model()