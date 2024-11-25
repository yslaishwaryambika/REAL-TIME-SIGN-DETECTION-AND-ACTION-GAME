import os
import time

def run_system():
    print("Starting ASL Recognition System...")
    
    # 1. Run preprocessing
    print("\n1. Starting preprocessing...")
    from preprocess import DataProcessor
    
    data_path = 'D:/ASL_1/Gesture Image Data'
    processor = DataProcessor(data_path)
    h5_path = processor.process_dataset()
    
    print("\nPreprocessing completed!")
    time.sleep(2)
    
    # 2. Train model
    print("\n2. Starting model training...")
    from model import ASLModel
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test, gesture_classes = DataProcessor.load_data(h5_path)
    
    # Initialize and train model
    asl_model = ASLModel(num_classes=len(gesture_classes))
    asl_model.build_model()
    asl_model.train(X_train, y_train, X_test, y_test)
    
    print("\nModel training completed!")
    time.sleep(2)
    
    # 3. Run game
    print("\n3. Starting game...")
    from game import ASLGame
    import h5py
    
    # Load model and gesture classes
    model_path = 'models/asl_model.h5'
    with h5py.File('processed_data/processed_data.h5', 'r') as f:
        gesture_classes = list(f['gesture_classes'][:])
    
    # Create and run game
    game = ASLGame(model_path, gesture_classes)
    game.run_game()

if __name__ == "__main__":
    try:
        run_system()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()