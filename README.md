# Project 2: MI3 Group5
MI3 Data Analysis for Project 2 Group 5 for DS 4002

Team Members:
Minh Nguyen(hvn9qwn): Group Leader,
Sally Sydnor(srs8yy),
David Bergman(dtb9de),


## Repository Contents

This repository contains 2 markdown files: README.md and LICENSE.md, as well as 3 folders: SRC, DATA, and Project2_images. The README.md file contains information about the contents of the repo as well as explanations for the src, data, and figures folders. LICENSE.md contains an MIT license for our work. The SRC folder contains the main code file for our project. More information about how the code works will be provided in the next section of this document. The data folder contains instructions on how to download the data file used for this project. A data dictionary is provided in the data section of this readme. The Project2_images folder will contain all of the graphics generated from this project. A description of each figure is provided in the figures section of the readme. Lastly, all of our references will be displayed in the references section of this readme.

## SRC

### Installation/Building of Code

#### Implement the following steps to install and build the code:
1. Importing packages
```{r}
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
drive.mount('/content/drive')
```
2. Importing data
```{r}
weather = "drive/MyDrive/ColabNotebooks/weather"
type(weather)
img_height = 180
img_width = 180

import os

# Define the URL of your dataset (change to your dataset's URL)
dataset_url = weather

# Define the local directory to save and extract the dataset
data_dir = os.path.join(os.path.dirname(weather), 'weather')

import pathlib
import numpy as np
data_dir2 = pathlib.Path(weather)
```
### Code Usage

Producing weather prediction model:

1. Generating Image Data
```{r}
image_size = (128, 128)  # Specify the desired image size
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill mode for data augmentation
)
train_datagen

datagen = ImageDataGenerator(
    rotation_range=40,       # Rotate images up to 40 degrees
    width_shift_range=0.2,   # Shift width up to 20% of the image width
    height_shift_range=0.2,  # Shift height up to 20% of the image height
    shear_range=0.2,         # Shear transformations
    zoom_range=0.2,          # Zoom in up to 20%
    horizontal_flip=True,    # Flip horizontally
    fill_mode='nearest'      # Fill in missing pixels with the nearest available pixel
)
```
2. Load and Preprocess Data
```{r}
train_data = tf.keras.utils.image_dataset_from_directory(
  data_dir2,
  validation_split=0.2, # splitting 80/20
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=30)
validation_data = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=30)
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir2,
  validation_split=0.2, # splitting 80/20
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=30)

train_ds2 = tf.keras.utils.image_dataset_from_directory(
  data_dir2,
  validation_split=0.15, # splitting 80/20
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=35)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=30)

val_ds2 = tf.keras.utils.image_dataset_from_directory(
  data_dir2,
  validation_split=0.15,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=35)
```
3. Creating Predictive Model
```{r}
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model2 = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Added softmax activation for classification
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
4. Running the Model
```{r}
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

epochs = 10
history2 = model2.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```
5. Creating Analysis Plots
```{r}
import matplotlib.pyplot as plt

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Model 1 Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 0].set_title('Model 1 Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

# Model 1 Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0, 1].set_title('Model 1 Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()

# Model 2 Loss
axes[1, 0].plot(history2.history['loss'], label='Train Loss')
axes[1, 0].plot(history2.history['val_loss'], label='Validation Loss')
axes[1, 0].set_title('Model 2 Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()

# Model 2 Accuracy
axes[1, 1].plot(history2.history['accuracy'], label='Train Accuracy')
axes[1, 1].plot(history2.history['val_accuracy'], label='Validation Accuracy')
axes[1, 1].set_title('Model 2 Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()

# Adjust the layout to prevent overlapping
plt.tight_layout()

# Display the plots
plt.show()
```
6. Testing Model Accuracy
```{r}
import tensorflow as tf
import numpy as np
import pathlib

# Define the paths to the test images
test_paths = [
    "drive/MyDrive/ColabNotebooks/weather/cloudy291.jpg",
    "drive/MyDrive/ColabNotebooks/weather/shine40.jpg",
    "drive/MyDrive/ColabNotebooks/weather/rain5.jpg",
    "drive/MyDrive/ColabNotebooks/weather/cloudy290.jpg",
    "drive/MyDrive/ColabNotebooks/weather/rain8.jpg"
]

# Initialize variables to accumulate scores
total_scores = 0.0
num_tests = 0

# Iterate through the test images
for test_path in test_paths:
    test_path = pathlib.Path(test_path)

    img = tf.keras.utils.load_img(test_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    total_scores += score
    num_tests += 1
    print(
        "Image {} most likely belongs to {} with a {:.2f} percent confidence."
        .format(test_path, class_names[np.argmax(score)], 100 * np.max(score))
    )
total_scores

# Calculate the average confidence
average_score = total_scores / num_tests
average_score
```
## Data

| Variable    | Variable Type | Description                                            |
| ----------- | ------------- | -------------------------------------------------------|
| Image       | jpg           | Image of a weather condition (labeled in image name)   |


Data file can be downloaded through this Mendeley Data link:
https://data.mendeley.com/datasets/4drtyfjtfy/1 


## Figures

### Project 2 Figures Table of Contents
| Figure Name      | Description |
| ----------- | ----------- |
| Image_Set_1.png | First sample run of predictive model, contains 9 images and its predictions, all the predictions were correct|
| Image_Set_2.png | Second sample run of predictive model, contains 9 images and its predictions, all the predictions were correct|
| Image_Set_3.png | Third sample run of predictive model, contains 9 images and its predictions, all the preditions were correct| 
| Training_Validation_Graph.png | Learning curves for the predictive model, created by plotting training and validation errors (losses) and accuracies against the number of epochs|
| Model Comparison.png | Learning curves showing the loss and accuracy for our predictive models|

View figures here: https://github.com/dtb9de/DS4002P2/tree/main/Project2_Images

Model Values
Model: "sequential"


## References
[1] Hannah Ritchie, Max Roser and Pablo Rosado (2020) - "CO₂ and Greenhouse Gas Emissions". Published online at OurWorldInData.org. Retrieved from: https://ourworldindata.org/co2-and-greenhouse-gas-emissions 

[2] “The Science of Climate Change,” The Science of Climate Change | The world is warming Wisconsin DNR, https://dnr.wisconsin.gov/climatechange/science (accessed Nov. 8, 2023). 

### Previous Submissions
MI1:  
MI2:  

