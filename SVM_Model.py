import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

class TBClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def load_images(self, folder_path):
        images = []
        labels = []

        for label, class_folder in enumerate(os.listdir(folder_path)):
            class_path = os.path.join(folder_path, class_folder)

            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error loading image: {img_path}")
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
                    img = cv2.resize(img, (100, 100))  # Resize images if needed
                    images.append(img.flatten())
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing image: {img_path}. Exception: {e}")

        return np.array(images), np.array(labels)

    def train(self, folder_path):
        # Load images
        images, labels = self.load_images(folder_path)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train SVM model
        self.model = svm.SVC(kernel='linear')
        self.model.fit(X_train_scaled, y_train)

        # Evaluate accuracy
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

    def save_model(self, filepath):
        if self.model:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("Model not trained yet.")

    def save_scaler(self, filepath):
        if self.scaler:
            joblib.dump(self.scaler, filepath)
            print(f"Scaler saved to {filepath}")
        else:
            print("Scaler not initialized yet.")

# Example usage:
classifier = TBClassifier()
classifier.train('D:/archive/TB Chest Radiography Database/TB Chest Radiography Database/Cleaned Data')
classifier.save_model('tb_classifier_model.pkl')
classifier.save_scaler('scaler.pkl')
