#!/usr/bin/env python
# coding: utf-8

# # **CNN LC phase classifier**


from google.colab import drive
drive.mount('/content/drive')

file_title = "First CNN"

import numpy as np
import tensorflow as tf
from keras import Model
from keras import layers
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

# Get necessary scripts from google drive
import sys
sys.path.insert(0,"/content/drive/MyDrive/")
from Model_metric_plotter_saver import (plot_metrics, 
                                        save_history_to_csv, 
                                        confusion_matrix_plot)

# Define where images are stored
img_directory = "/content/drive/My Drive/I-N-Chol-Sm_dataset-balanced"

train_dir = img_directory + "/Train"
val_dir = img_directory + "/Val"

# These lines of code allow for the datasets to be loaded in batch_size at
# a time from files
image_size=(256,256)
train_dataset = image_dataset_from_directory(train_dir,
                            labels="inferred",
                            label_mode="categorical",
                            color_mode="grayscale",
                            batch_size=64,
                            image_size=image_size,
                            shuffle=True
                        )
val_dataset = image_dataset_from_directory(val_dir,
                            labels="inferred",
                            label_mode="categorical",
                            color_mode="grayscale",
                            batch_size=64,
                            image_size=image_size,
                            shuffle=False # Set to false so confusion matrix
                                          # can be plotted
                        )

# These next steps enable the dataset to be stored in cache (if it can fit)
# meaning that after data is loaded in on the first epoch, it will
# be loaded in much faster
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# Define model layers
# Input/preprocessing layers
input_layer = layers.Input(shape = (256,256,1), name="Input")
X = layers.experimental.preprocessing.Rescaling(1./255, 
                                                name="Normalize")(input_layer)
# Augmentation layers
X = layers.experimental.preprocessing.RandomFlip()(X)
X = layers.experimental.preprocessing.RandomRotation(0.2)(X) # rotate by 0.2*2pi
X = layers.experimental.preprocessing.RandomZoom(0.2)(X)

# Conv layers
X = layers.Conv2D(16, kernel_size=(5,5), activation="relu", name="Conv1")(X)
X = layers.MaxPooling2D(2,2, name="MaxPool1")(X)

X = layers.Conv2D(32, kernel_size=(3,3), activation="relu", name="Conv2")(X)
X = layers.MaxPooling2D(2,2, name="MaxPool2")(X)

X = layers.Conv2D(64, kernel_size=(3,3), activation="relu", name="Conv3")(X)
X = layers.MaxPooling2D(2,2, name="MaxPool3")(X)

X = layers.Conv2D(128, kernel_size=(3,3), activation="relu", name="Conv4")(X)
X = layers.MaxPooling2D(2,2, name="MaxPool4")(X)

# Fully connected layers
X = layers.Flatten()(X)

X = layers.Dense(units=128, activation="relu", name="FC5")(X)
X = layers.Dropout(0.5)(X)

X = layers.Dense(units=254, activation="relu", name="FC6")(X)
X = layers.Dropout(0.5)(X)

X = layers.Dense(units=64, activation="relu", name="FC7")(X)
X = layers.Dropout(0.5)(X)

# Output layer 
X = layers.Dense(units=4, activation="softmax")(X)

# Save a diagram of the model
model = Model(input_layer, X, name=file_title)
model.summary()
model_save_path = "/content/drive/My Drive/" + file_title + ".png"
plot_model(model, to_file=model_save_path, 
           show_shapes=True, show_layer_names=True)

metrics = ["accuracy"] # Accuracy is the metric of interest while trainin the
                       # model
# This is where to save a trained model
model_save_dir = "/content/drive/My Drive/" + file_title + "_saved_model"

model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])
# Use a checkpoint to save a most at the epoch of best validation accuracy
checkpoint = ModelCheckpoint(model_save_dir, 
                             monitor = "val_accuracy", 
                             save_best_only = True, mode="max")
# Train the model for 60 epochs and return the data each epoch in history
# for plotting later
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=60,
                    callbacks=[checkpoint],
                    verbose=1
                   )

save_dir = "/content/drive/My Drive/" + file_title

metrics = ["loss", "accuracy"]
# Plot the training history of the model and save
training_plot_save_path = save_dir + "_training_history"
plot_metrics(history, training_plot_save_path, metrics)

# Save training history to csv
csv_data_save_path = save_dir + "_training_history.csv"
save_history_to_csv(history, csv_data_save_path, metrics)

# Evaluate model on validation data and save the confusion matrix
# Get best saved model
trained_model = tf.keras.models.load_model(model_save_dir)
# Get predictions and true labels
predictions = trained_model.predict(val_dataset, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(np.concatenate([label for image, label in val_dataset], 
                                  axis=0), axis=1)
# Plot and save the confusion matrix
target_names = ["Isotropic", "Nematic", "Cholesteric", "Smectic"]
confusion_mat_save_dir = save_dir + "_confusion_matrix"
confusion_matrix_plot(y_true, y_pred, target_names,
                      save=True, save_path=confusion_mat_save_dir)

# Applying the trained model to test video
from lc_video_phase_labeller import PhaseLabeller

model_load_dir = "C:/Users/Jason/Documents/University/Year_4/Saved_model/"
phase_list = ["Isotropic", "Nematic", "Cholesteric", "Smectic"]
vid_file = "N-I/"
vid_name = "nematic 5cb on glycerol-2-Yellow_35.6C_cooling"
vid_path = "C:/Users/Jason/Documents/" + vid_file + vid_name + ".avi"
save_dir = "C:/Users/Jason/Documents/University/"
vid_save_path = save_dir + vid_name + "_labelled.avi" 
start_temp = 35.6
end_temp = None
temp_rate = 0.1 # rate of temperature decrease in seconds

phase_vid_labeller = PhaseLabeller()
phase_vid_labeller.get_model(model_load_dir, phase_list)

phase_vid_labeller.label_video(vid_path,
                               vid_save_path,
                               start_temp,
                               end_temp=end_temp,
                               temp_rate_per_sec=temp_rate)

# Save labelling prediction and temperature data in csv
save_path = save_dir + vid_name + ".csv"
phase_vid_labeller.to_csv(save_path)

# Plot and save the prediction confidence against temperature graph
save_path = save_dir + vid_name + ".png"
#phase_plot_list = ["Isotropic", "Nematic", "Cholesteric", "Smectic"]
phase_plot_list = ["Isotropic", "Nematic"]
#color_list = ["g", "b", "r", "k"]
color_list = ["g", "b"]
phase_vid_labeller.conf_temp_plot(phase_plot_list,
                                  color_list,
                                  save_path)


save_path = save_dir + vid_name + "-zoomed_in.png"
phase_vid_labeller.conf_temp_plot(phase_plot_list,
                                  color_list,
                                  save_path,
                                  start_temp=40.5,
                                  end_temp=41.5)