# Handwritten Character Recognition using CNN

This project recognizes handwritten alphabet characters (A-Z) using a Convolutional Neural Network (CNN) implemented in Python with TensorFlow and Keras.

## Project Structure

- `handwritten-character-recognition.ipynb` – Jupyter Notebook with the full implementation  
- `handwritten-character-recognition.py` – Python script version of the notebook   
- `README.md` – Project description  

## Features

- Trains a CNN on the A-Z Handwritten Data CSV dataset  
- Visualizes the distribution of characters and sample images  
- Uses multiple convolutional and dense layers for accurate predictions  
- Can predict both test set and external images  
- Saves the trained model for future predictions  

## Dependencies

Install required Python packages:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow keras
