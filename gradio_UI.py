import gradio as gr
import joblib
import cv2
import numpy as np

class TBClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('D:/power BI project/diff_project_topic/templates/scaler.pkl')

    def preprocess(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (100, 100))
        img_flatten = img_resized.flatten()
        img_scaled = self.scaler.transform([img_flatten])
        img_reshaped = img_scaled.reshape(1, -1)
        return img_reshaped

    def predict(self, img):
        preprocessed_img = self.preprocess(img)
        prediction = self.model.predict(preprocessed_img)
        return 'Normal' if prediction[0] == 0 else 'TB Infected'

# Load SVM model and scaler
classifier = TBClassifier('D:/power BI project/diff_project_topic/templates/tb_classifier_model.pkl')

# Create Gradio interface
iface = gr.Interface(fn=classifier.predict, inputs="image", outputs="text",
                     title="TB Classification", description="Upload an X-ray image and classify it as Normal or TB Infected.")
iface.launch(share=True)


