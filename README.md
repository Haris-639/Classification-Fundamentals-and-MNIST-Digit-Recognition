# MNIST Digit Recognition with Scikit-learn

This project builds and deploys a digit classifier using the MNIST dataset and two ML models: SGDClassifier and RandomForestClassifier.

## Features
- Load and preprocess the MNIST dataset
- Train classifiers: SGD (hinge loss) and Random Forest
- Evaluate using confusion matrix and classification report
- Visualize common misclassifications
- Improve performance using data augmentation
- Deploy as an interactive Gradio app

## Setup Instructions

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the notebook train the models
3. Run python script and lauch it to run gradio code (app.py)
   ```bash
    python app.py
4.After this a link will be shown run it on your browser and a page will be shown where you can upload test image and select the model to check the perdiction.
