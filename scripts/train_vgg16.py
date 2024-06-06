import os
import cv2
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import re
import json

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message=re.escape('Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor.'))

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Configure TensorFlow to use as much GPU memory as possible
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

## Definitions ##
# Data path
data_dir = r'C:\Users\ertug\Desktop\Folder\makine_ögrenmesi_proje\images'
model_dir = r'C:\Users\ertug\Desktop\Folder\makine_ögrenmesi_proje\models'
class_indices_path = os.path.join(model_dir, 'class_indices.json')
feature_maps_dir = os.path.join(model_dir, 'feature_maps')

# Verify that the directories exist
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Model directory created at: {model_dir}")
if not os.path.exists(feature_maps_dir):
    os.makedirs(feature_maps_dir)
    print(f"Feature maps directory created at: {feature_maps_dir}")

# Data augmentation and normalization
print("Setting up data augmentation and normalization...")
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of data will be used for validation
)

# Create data generators
print("Creating data generators...")
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# VGG16 Model Definition
print("Defining the VGG16 model...")
input_layer = Input(shape=(224, 224, 3))

x = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(input_layer)
x = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Flatten()(x)
x = Dense(4096, activation="relu")(x)
x = Dense(4096, activation="relu")(x)
output_layer = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Model Summary
print("Model summary:")
model.summary()

# Model Compilation
print("Compiling the model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
print("Starting model training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=25,
    validation_data=val_generator
)

# Save the model
model_path = os.path.join(model_dir, 'qedy_vgg16_model.keras')
print(f"Saving the model as '{model_path}'...")
model.save(model_path)

# Save class indices
class_indices = train_generator.class_indices
with open(class_indices_path, 'w') as f:
    json.dump(class_indices, f)
print(f"Class indices saved to {class_indices_path}")

# Evaluate the model
print("Evaluating the model on validation data...")
loss, accuracy = model.evaluate(val_generator, steps=val_generator.samples // val_generator.batch_size)
print(f'Validation loss: {loss:.4f}')
print(f'Validation accuracy: {accuracy:.4f}')

# Plot and save training & validation accuracy values
accuracy_plot_path = os.path.join(model_dir, 'accuracy_plot.png')
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(accuracy_plot_path)
print(f"Accuracy plot saved to {accuracy_plot_path}")

# Plot and save training & validation loss values
loss_plot_path = os.path.join(model_dir, 'loss_plot.png')
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(loss_plot_path)
print(f"Loss plot saved to {loss_plot_path}")

# Visualizing the feature maps
def plot_feature_maps(model, img_array, save_dir):
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
                if channel_image.std() == 0:
                    continue
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= (channel_image.std() + 1e-5)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        
        # Save the feature map
        feature_map_path = os.path.join(save_dir, f"{layer_name}_feature_map.png")
        try:
            plt.savefig(feature_map_path)
        except SystemError as e:
            print(f"Error saving feature map for layer {layer_name}: {e}")
        plt.close()
        print(f"Feature map saved to {feature_map_path}")

# Plot and save feature maps from a validation image
print("Generating feature maps from a validation image...")
validation_image_batch, _ = next(val_generator)  # Get a batch of validation images
sample_image_array = validation_image_batch[0:1]  # Use the first image from the batch
plot_feature_maps(model, sample_image_array, feature_maps_dir)
