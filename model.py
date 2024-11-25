import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from preprocess import DataProcessor

class ASLModel:
    def __init__(self, num_classes, img_size=50):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None

    def build_model(self):
        """Build and compile the CNN model"""
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(self.img_size, self.img_size, 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Dropout(0.25),

            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Dropout(0.25),

            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Dropout(0.25),

            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile model with optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(self.model.summary())
        return self.model

    def train(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        """Train the model with callbacks"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Create callbacks
        callbacks = [
            # Save best model
            ModelCheckpoint(
                'models/best_model.keras',  # Changed extension to .keras
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train model
        print("\nStarting model training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        self.model.save('models/final_model.keras')  # Changed extension to .keras
        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        scores = self.model.evaluate(X_test, y_test, verbose=1)
        print(f'\nTest accuracy: {scores[1]*100:.2f}%')
        return scores

    def plot_training_history(self):
        """Plot training history"""
        # Create directory for plots
        os.makedirs('plots', exist_ok=True)

        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png')
        plt.close()

def main():
    # Load processed data
    print("Loading preprocessed data...")
    h5_path = 'processed_data/processed_data.h5'
    X_train, X_test, y_train, y_test, gesture_classes = DataProcessor.load_data(h5_path)

    # Initialize and train model
    print("\nInitializing model...")
    asl_model = ASLModel(num_classes=len(gesture_classes))
    asl_model.build_model()

    # Train model
    print("\nTraining model...")
    asl_model.train(X_train, y_train, X_test, y_test)

    # Evaluate model
    print("\nEvaluating model...")
    asl_model.evaluate(X_test, y_test)

    # Plot training history
    print("\nPlotting training history...")
    asl_model.plot_training_history()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)
    
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()