# app.py

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import joblib

# Load both trained models
rf_model = joblib.load("random_forest_model.pkl")
sgd_model = joblib.load("sgd_model.pkl")

# Prediction function
def predict_digit(img, model_choice):
    if img is None:
        return "No image provided"

    # Convert to grayscale, resize, invert colors (white on black), normalize
    img = Image.fromarray(img).convert("L")
    img = ImageOps.invert(img).resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 784)

    # Choose model
    if model_choice == "Random Forest":
        prediction = rf_model.predict(img_array)[0]
    else:
        prediction = sgd_model.predict(img_array)[0]

    return int(prediction)

# Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=[
        gr.Image(image_mode="L", type="numpy"),
        gr.Radio(["Random Forest", "SGD"], label="Choose Model")
    ],
    outputs="label",
    title="MNIST Digit Classifier",
    description="Draw a digit (0–9), choose a model, and get a prediction"
)

if __name__ == "__main__":
    interface.launch(share=True)
