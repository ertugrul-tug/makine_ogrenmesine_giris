import os
import keras
from keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import json

# Define paths
model_dir = r'C:\Users\ertug\Desktop\Folder\makine_ögrenmesi_proje\models'
model_path = os.path.join(model_dir, 'qedy_vgg16_model.keras')
class_indices_path = os.path.join(model_dir, 'class_indices.json')
sample_image_path = r'C:\Users\ertug\Desktop\Folder\makine_ögrenmesi_proje\sample.jpg'

# Load the trained model
print("Loading the model...")
model = load_model(model_path)
print("Model loaded successfully.")

# Load class indices
print("Loading class indices...")
if not os.path.exists(class_indices_path):
    raise FileNotFoundError(f"Class indices file not found: {class_indices_path}")

with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
class_indices = {v: k for k, v in class_indices.items()}  # Invert the class_indices dictionary

# Load and preprocess the sample image
print("Loading and preprocessing the sample image...")
if not os.path.exists(sample_image_path):
    raise FileNotFoundError(f"Sample image not found: {sample_image_path}")

sample_image = load_img(sample_image_path, target_size=(224, 224))
sample_image_array = img_to_array(sample_image) / 255.0
sample_image_array = np.expand_dims(sample_image_array, axis=0)

# Make predictions
print("Making predictions on the sample image...")
predictions = model.predict(sample_image_array)
predicted_class = np.argmax(predictions, axis=1)

# Display the image and prediction
print("Displaying the sample image with the prediction...")
plt.imshow(sample_image)
plt.title(f'Predicted Class: {class_indices[predicted_class[0]]}')
plt.axis('off')
plt.show()

# Visualizing the feature maps
def plot_feature_maps(model, img_array):
    # Ensure the model has been called to set input shape
    _ = model.predict(img_array)
    
    # Extracts the outputs of the top 8 layers:
    layer_outputs = [layer.output for layer in model.layers[:8]] 
    # Creates a model that will return these outputs, given the model input:
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    # Returns a list of five Numpy arrays: one array per layer activation
    activations = activation_model.predict(img_array)
    
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] # The feature map has shape (1, size, size, n_features)
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

plot_feature_maps(model, sample_image_array)
