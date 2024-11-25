import os
import cv2
import numpy as np
import h5py
import mediapipe as mp
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path, output_path='processed_data', img_size=50):
        self.data_path = data_path
        self.output_path = output_path
        self.img_size = img_size
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)

    def process_image(self, img):
        """Process single image with hand detection"""
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            h, w = img.shape[:2]
            landmarks = results.multi_hand_landmarks[0]
            
            # Get hand bounding box
            x_min, y_min = w, h
            x_max = y_max = 0
            
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            
            if x_min < x_max and y_min < y_max:
                hand_region = img[y_min:y_max, x_min:x_max]
                return cv2.resize(hand_region, (self.img_size, self.img_size))
        
        return cv2.resize(img, (self.img_size, self.img_size))

    def process_batch(self, image_paths, labels):
        """Process a batch of images"""
        batch_size = len(image_paths)
        processed_data = np.zeros((batch_size, self.img_size, self.img_size, 3), dtype='float32')
        valid_indices = []
        valid_labels = []
        
        for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
            img = cv2.imread(img_path)
            if img is not None:
                processed_img = self.process_image(img)
                processed_data[idx] = processed_img
                valid_indices.append(idx)
                valid_labels.append(label)
        
        return processed_data[valid_indices] / 255.0, valid_labels

    def process_dataset(self, batch_size=1000):
        """Process the entire dataset in batches"""
        print("Starting dataset processing...")
        
        h5_path = os.path.join(self.output_path, 'processed_data.h5')
        gesture_classes = sorted(os.listdir(self.data_path))
        
        with h5py.File(h5_path, 'w') as hf:
            # Create extensible datasets
            data_set = hf.create_dataset(
                'data',
                shape=(0, self.img_size, self.img_size, 3),
                maxshape=(None, self.img_size, self.img_size, 3),
                chunks=True,
                dtype='float32'
            )
            
            label_set = hf.create_dataset(
                'labels',
                shape=(0, len(gesture_classes)),
                maxshape=(None, len(gesture_classes)),
                chunks=True,
                dtype='float32'
            )
            
            # Store gesture classes
            dt = h5py.special_dtype(vlen=str)
            gesture_dataset = hf.create_dataset('gesture_classes', (len(gesture_classes),), dtype=dt)
            gesture_dataset[:] = gesture_classes
            
            total_processed = 0
            
            for idx, gesture in enumerate(gesture_classes):
                print(f"\nProcessing gesture: {gesture}")
                gesture_path = os.path.join(self.data_path, gesture)
                image_files = [f for f in os.listdir(gesture_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                for i in range(0, len(image_files), batch_size):
                    batch_files = image_files[i:i + batch_size]
                    batch_paths = [os.path.join(gesture_path, f) for f in batch_files]
                    batch_labels = [idx] * len(batch_files)
                    
                    processed_data, valid_labels = self.process_batch(batch_paths, batch_labels)
                    
                    if len(valid_labels) > 0:
                        # Convert labels to one-hot encoding
                        processed_labels = to_categorical(valid_labels, num_classes=len(gesture_classes))
                        
                        # Resize dataset and add new data
                        current_size = data_set.shape[0]
                        new_size = current_size + len(processed_data)
                        
                        data_set.resize((new_size, self.img_size, self.img_size, 3))
                        label_set.resize((new_size, len(gesture_classes)))
                        
                        data_set[current_size:new_size] = processed_data
                        label_set[current_size:new_size] = processed_labels
                        
                        total_processed += len(processed_data)
                        print(f"Processed {total_processed} images total")

        print("\nDataset processing completed!")
        return h5_path

    @staticmethod
    def load_data(h5_path, test_size=0.2):
        """Load processed dataset and split into train/test sets"""
        with h5py.File(h5_path, 'r') as hf:
            data = hf['data'][:]
            labels = hf['labels'][:]
            gesture_classes = list(hf['gesture_classes'][:])
        
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, gesture_classes

def main():
    data_path = 'D:/ASL_1/Gesture Image Data'
    processor = DataProcessor(data_path)
    
    # Process dataset
    h5_path = processor.process_dataset()
    
    # Load and split data
    X_train, X_test, y_train, y_test, gesture_classes = DataProcessor.load_data(h5_path)
    print("\nDataset Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {len(gesture_classes)}")

if __name__ == "__main__":
    main()