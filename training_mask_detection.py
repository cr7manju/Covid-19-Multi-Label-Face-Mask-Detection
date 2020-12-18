# -*- coding: utf-8 -*-
##########################################################################################
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import os
import argparse
import seaborn as sns

# Hyper-parameter tuning
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from kerastuner.engine.hyperparameters import HyperParameters
##########################################################################################

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, help="path to input dataset")
parser.add_argument('-o', '--output', required=True, help="path to output each epoch")
parser.add_argument("-p1", "--confusion_matrix", type=str, default="confusion_matrix.png", help="path to output loss plot")
parser.add_argument("-p2", "--accuracy_loss_plot", type=str, default="accuracy_loss_plot.png", help="path to output accuracy plot")
parser.add_argument("-vgg19", "--model1", type=str, default="vgg19_mask_detection.hdf5", help="path to output face mask detector VGG19 model")
parser.add_argument("-mv2", "--model2", type=str, default="mv2_mask_detection.hdf5", help="path to output face mask detector MobileNetV2 model")
parser.add_argument("-effb7", "--model3", type=str, default="effb7_mask_detection.hdf5", help="path to output face mask detector MobileNetV2 model")

args = vars(parser.parse_args())
###########################################################################################
# Provide the path of all the images and the categories it needs to be categorized
data_directory = args["dataset"]
#data_directory = list(paths.list_images(args["dataset"]))
output_directory = args["output"]
#output_directory = list(paths.list_images(args["output"]))
# Gets the folder names inside the given directory
categories = os.listdir(data_directory)
# Store the categories as labels with 0,1,2
labels = [i for i in range(len(categories))]
# create a dictonary variable and storing category and label information as key-value pair
#label_dict = dict(zip(categories, labels))
label_dict = {}
for c in categories:
    if c == "No Mask":
        label_dict[c] = 0
    elif c == "Wrong Mask":
        label_dict[c] = 1   
    elif c == "Mask":
        label_dict[c] = 2
# Store number of categories based on number of labels
noc = len(labels)
print ("Categories: ", categories)
print ("Labels: ", labels )
print ("Category and its Label information: ", label_dict)
###########################################################################################

# Converting the image into array and normalizing the image using preprocess block
def preprocessing_image(preprocess_type, directory, category):
  image_size = 224
  mv2_data = []
  mv2_labels = []
  vgg19_data = []
  vgg19_labels = []
  effb7_data = []
  effb7_labels = []
  print("[INFO] loading images...")
  if preprocess_type.strip().upper() == "MV2":
    for category in categories:
        path = os.path.join(data_directory, category)
      # load the input image (224*224) and preprocess it
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(image_size, image_size))
            image = img_to_array(image) # converted to (224,224,3) dimensions
            #print("image: ", image)
            #print("image shape: ", image.shape)
            image = mv2_preprocess(image) # Resizing scaled to -1 to 1 
            #print("preprocessed image: ", image)
            #print("preprocessed image shape: ", image.shape)      
      # update the data and labels lists, respectively
            mv2_data.append(image)
            mv2_labels.append(label_dict[category])
    # Saving the image data and target data using numpy for backup
    np.save(os.path.join(output_directory, "mv2_data"), mv2_data)
    np.save(os.path.join(output_directory, "mv2_labels"), mv2_labels)
  elif preprocess_type.strip().upper() == "VGG19":
    for category in categories:
        path = os.path.join(data_directory, category)
      # load the input image (224*224) and preprocess it
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(image_size, image_size))
            image = img_to_array(image) # converted to (224,224,3) dimensions
            #print("image: ", image)
            #print("image shape: ", image.shape)
            image = vgg19_preprocess(image) # Resizing scaled to -1 to 1 
            #print("preprocessed image: ", image)
            #print("preprocessed image shape: ", image.shape)      
      # update the data and labels lists, respectively
            vgg19_data.append(image)
            vgg19_labels.append(label_dict[category])
    # Saving the image data and target data using numpy for backup
    np.save(os.path.join(output_directory, "vgg19_data"), vgg19_data)
    np.save(os.path.join(output_directory, "vgg19_labels"), vgg19_labels)
  elif preprocess_type.strip().upper() == "EFFB7":
        for category in categories:
            path = os.path.join(data_directory, category)
          # load the input image (224*224) and preprocess it
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                image = load_img(img_path, target_size=(image_size, image_size))
                image = img_to_array(image) # converted to (224,224,3) dimensions
                #print("image: ", image)
                #print("image shape: ", image.shape)
                image = effnet_preprocess(image) # Resizing scaled to -1 to 1 
                #print("preprocessed image: ", image)
                #print("preprocessed image shape: ", image.shape)      
          # update the data and labels lists, respectively
                effb7_data.append(image)
                effb7_labels.append(label_dict[category])
        # Saving the image data and target data using numpy for backup
        np.save(os.path.join(output_directory, "effb7_data"), effb7_data)
        np.save(os.path.join(output_directory, "effb7_labels"), effb7_labels)
    
  print("Images loaded and saved Successfully ")
###########################################################################################  

# preprocessing the images using Mobilenetv2 and save it
preprocessing_image("MV2", data_directory, categories)

# loading the saved numpy arrays of mobilenetv2 preprocessed images
mv2_data = np.load(os.path.join(output_directory, "mv2_data.npy"))
mv2_labels = np.load(os.path.join(output_directory, "mv2_labels.npy"))
########################################################################################

# Bar plot to see the count of images in various categories
unique, counts = np.unique(mv2_labels, return_counts=True)
category_count = dict(zip(unique, counts))
temp_df = pd.DataFrame({'Labels': list(category_count.keys()), 'Count': list(category_count.values())})
#temp_df['Labels'] = pd.DataFrame(list(category_count.keys()))
#temp_df['Count'] = pd.DataFrame(list(category_count.values()))
temp_df.loc[temp_df['Labels'] == 0, 'Categories'] = list(label_dict.keys())[list(label_dict.values()).index(0)]
temp_df.loc[temp_df['Labels'] == 1, 'Categories'] = list(label_dict.keys())[list(label_dict.values()).index(1)]
temp_df.loc[temp_df['Labels'] == 2, 'Categories'] = list(label_dict.keys())[list(label_dict.values()).index(2)]
plt.barh(temp_df.Categories, temp_df.Count, color='rgbkymc')
plt.ylabel("Various Categories")
plt.xlabel("Count")
plt.title("Bar plot to see the count of images based on categories")
for index, value in enumerate(temp_df.Count):
    plt.text(value, index, str(value))
plt.show()
##########################################################################################

# Print some of the random images with labelled data
# Use subplots to show images with the categories it belongs to
num_rows, num_cols = noc, noc+3
nrow = 10
ncol = 3

f, ax = plt.subplots(num_rows, num_cols, figsize=(12,7))
for r in range(num_rows):
    temp = np.where(mv2_labels==r)[0][0]
    print(temp)
    for c in range(num_cols):
        image_index = temp + c
        #print(image_index)
        ax[r,c].axis("off")
        ax[r,c].imshow( mv2_data[image_index])
        ax[r,c].set_title(list(label_dict.keys())[list(label_dict.values()).index(mv2_labels[image_index])])
        plt.subplots_adjust(wspace=None, hspace=None)
f.suptitle("Images after pre processed using Mobilenet V2")
plt.show()
plt.close()
############################################################################################

# convert output array of labeled data(from 0 to nb_classes - 1) to one-hot vector
mv2_labels = to_categorical(mv2_labels)
print ("Shape of mv2 data: ", mv2_data.shape)
print ("Shape of mv2 labels: ", mv2_labels.shape)
###########################################################################################

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.1,
	horizontal_flip=True,
	fill_mode="nearest")
#############################################################################################

# Build Hyper Tunning model for MobileNetV2
def mv2_build_model(hp):
    baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = keras.layers.Flatten()(headModel)
    headModel = keras.layers.Dense(units = hp.Int(name = 'dense_units', min_value=64, max_value=256, step=16), activation = 'relu')(headModel)
    headModel = keras.layers.Dropout(hp.Float(name = 'dropout', min_value = 0.1, max_value = 0.5, step=0.1, default=0.5))(headModel)
    headModel = keras.layers.Dense(3, activation = 'softmax')(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
      layer.trainable = False
    model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
    return model

########################################################################################
# Stratified Split the data into 20% Test and 80% Traininng data
X_train,X_test,y_train, y_test = train_test_split(mv2_data, mv2_labels, test_size = 0.2, stratify = mv2_labels, random_state = 42)

print("MobilenetV2 Train data shape: ",X_train.shape)
print("MobilenetV2 Train label shape: ",y_train.shape)
print("MobilenetV2 Test data shape: ",X_test.shape)
print("MobilenetV2 Test label shape: ",y_test.shape)

np.save(os.path.join(output_directory, "X_test"), X_test)
np.save(os.path.join(output_directory, "y_test"), y_test)
##########################################################################################
# delete data and label variables to release ram
del mv2_data
del mv2_labels
##########################################################################################
# Hyperparameter tuning
print ("Search for best model fit for mobilenetv2...")
mv2_tuner_search = RandomSearch(mv2_build_model, objective = 'val_accuracy', max_trials =30, directory = output_directory, project_name = "MobileNetV2")
mv2_tuner_search.search(X_train, y_train, epochs = 3, validation_split = 0.2)

# Show a summary of the hyper parameter search output for 30 combination parameters
mv2_tuner_search.results_summary()
mv2_model = mv2_tuner_search.get_best_models(num_models=1)[0]
mv2_model.summary()
#########################################################################################
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath=args["model2"], monitor='val_loss', save_best_only=True)]

# Train neural network
mv2_history = mv2_model.fit(aug.flow(X_train, y_train, batch_size=64),
                      epochs=30, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=2, # Print description after each epoch
                      validation_data=(X_test, y_test)) # Data for evaluation

np.save(os.path.join(output_directory, 'mv2_history.npy'),mv2_history.history)
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(mv2_history.history) 
# or save to csv: 
hist_csv_file = os.path.join(output_directory, 'mv2_history.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
#############################################################################################
# preprocessing the images using VGG19 and save it
preprocessing_image("VGG19", data_directory, categories)

# loading the saved numpy arrays of vgg19 preprocessed images
vgg19_data = np.load(os.path.join(output_directory, "vgg19_data.npy"))
vgg19_labels = np.load(os.path.join(output_directory, "vgg19_labels.npy"))

# Check the data shape and labels after preprocessing of vgg19 image data
print("first 10 target values: ", vgg19_labels[1:10])
print("shape of data", vgg19_data[0].shape)
print("first image data information in array", vgg19_data[0])

# convert output array of labeled data(from 0 to nb_classes - 1) to one-hot vector
vgg19_labels = to_categorical(vgg19_labels)
print ("Shape of vgg19 data: ", vgg19_data.shape)
print ("Shape of vgg19 labels: ", vgg19_labels.shape)
###############################################################################################

# Build Hyper Tunning model for VGG19
def vgg19_build_model(hp):
    baseModel = VGG19(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = keras.layers.MaxPooling2D(pool_size=(5, 5))(headModel)
    headModel = keras.layers.Flatten()(headModel)
    headModel = keras.layers.Dense(units = hp.Int(name = 'dense_units', min_value=64, max_value=256, step=16), activation = 'relu')(headModel)
    headModel = keras.layers.Dropout(hp.Float(name = 'dropout', min_value = 0.1, max_value = 0.5, step=0.1, default=0.5))(headModel)
    headModel = keras.layers.Dense(3, activation = 'softmax')(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
      layer.trainable = False
    model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
    return model
#################################################################################################
# Stratified Split the data into 20% Test and 80% Traininng data
X2_train,X2_test,y2_train, y2_test = train_test_split(vgg19_data, vgg19_labels, test_size = 0.2, stratify = vgg19_labels, random_state = 42)

print("VGGNET19 Train data shape: ",X2_train.shape)
print("VGGNET19 Train label shape: ",y2_train.shape)
print("VGGNET19 Test data shape: ",X2_test.shape)
print("VGGNET19 Test label shape: ",y2_test.shape)

np.save(os.path.join(output_directory, "X2_test"), X2_test)
np.save(os.path.join(output_directory, "y2_test"), y2_test)
##############################################################################################
# delete data and label variables to release ram
del vgg19_data
del vgg19_labels
###########################################################################################
# Tuuning
print ("Search for best model fit for vgg19...")
vgg19_tuner_search = RandomSearch(vgg19_build_model, objective = 'val_accuracy', max_trials =30, directory = output_directory, project_name = "VGGNET19")
vgg19_tuner_search.search(X2_train, y2_train, epochs = 3, validation_split = 0.2)
# Show a summary of the hyper parameter search output for 100 combination parameters
vgg19_tuner_search.results_summary()
vgg19_model = vgg19_tuner_search.get_best_models(num_models=1)[0]
vgg19_model.summary()

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath=args["model1"], monitor='val_loss', save_best_only=True)]
###########################################################################################
# Train neural network
vgg19_history = vgg19_model.fit(aug.flow(X2_train, y2_train, batch_size=64),
                      epochs=30, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=2, # Print description after each epoch
                      validation_data=(X2_test, y2_test)) # Data for evaluation


np.save(os.path.join(output_directory, 'vgg19_history.npy'),vgg19_history.history)
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(vgg19_history.history) 
# or save to csv: 
hist_csv_file = os.path.join(output_directory, 'vgg19_history.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

############################################################################################
# preprocessing the images using EfficientNet  and save it
preprocessing_image("EFFB7", data_directory, categories)    
    
# loading the saved numpy arrays of EfficientNet preprocessed images
effb7_data = np.load(os.path.join(output_directory, "effb7_data.npy"))
effb7_labels = np.load(os.path.join(output_directory, "effb7_labels.npy"))

# Check the data shape and labels after preprocessing of efficentnet b7 image data
print("first 10 target values: ", effb7_labels[1:10])
print("shape of data", effb7_data[0].shape)
print("first image data information in array", effb7_data[0])

# convert output array of labeled data(from 0 to nb_classes - 1) to one-hot vector
effb7_labels = to_categorical(effb7_labels)
print ("Shape of efficient net data: ", effb7_data.shape)
print ("Shape of efficient net labels: ", effb7_labels.shape)
#######################################################################################

def effb7_build_model(hp):
    baseModel = EfficientNetB7(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = keras.layers.GlobalMaxPooling2D()(headModel)
    #keras.layers.MaxPooling2D(pool_size=(5, 5))(headModel)
    headModel = keras.layers.Flatten()(headModel)
    headModel = keras.layers.Dense(units = hp.Int(name = 'dense_units', min_value=64, max_value=1024, step=64), activation = 'relu')(headModel)
    headModel = keras.layers.Dropout(hp.Float(name = 'dropout', min_value = 0.4, max_value = 0.8, step=0.1, default=0.5))(headModel)
    headModel = keras.layers.Dense(3, activation = 'softmax')(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
      layer.trainable = False
    model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
    return model
##########################################################################################

# Stratified Split the data into 20% Test and 80% Traininng data for EFFB7 model
X3_train,X3_test,y3_train, y3_test = train_test_split(effb7_data, effb7_labels, test_size = 0.2, stratify = effb7_labels, random_state = 42)

print("efficient net Train data shape: ",X3_train.shape)
print("efficient net Train label shape: ",y3_train.shape)
print("efficient net Test data shape: ",X3_test.shape)
print("efficient net Test label shape: ",y3_test.shape)

np.save(os.path.join(output_directory, "X3_test"), X3_test)
np.save(os.path.join(output_directory, "y3_test"), y3_test)

##############################################################################################
# delete data and label variables to release ram
del effb7_data
del effb7_labels

#######################################################################################
#Tuning
print ("Search for best model fit for Effb7...")
effb7_tuner_search = RandomSearch(effb7_build_model, objective = 'val_accuracy', max_trials =30, directory = output_directory, project_name = "EfficientNet")
effb7_tuner_search.search(X3_train, y3_train, epochs = 3, validation_split = 0.2)

# Show a summary of the hyper parameter search output for 30 combination parameters
effb7_tuner_search.results_summary()

# Show the best parameters choosen model
effb7_model = effb7_tuner_search.get_best_models(num_models=1)[0]
effb7_model.summary()

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath=args["model3"], monitor='val_loss', save_best_only=True)]
########################################################################################################################################
# Train neural network with best model for 30 epochs
effb7_history = effb7_model.fit(aug.flow(X3_train, y3_train, batch_size=64),
                      epochs=30, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=2, # Print description after each epoch
                      validation_data=(X3_test, y3_test)) # Data for evaluation


np.save(os.path.join(output_directory, 'effb7_history.npy'),effb7_history.history)
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(effb7_history.history) 
# or save to csv: 
hist_csv_file = os.path.join(output_directory, 'effb7_history.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
###############################################################################
# Plot the loss and accuracy for both VGG19 and MobileNetV2 models
mv2_history_csv = pd.read_csv(os.path.join(output_directory, 'mv2_history.csv'))
vgg19_history_csv = pd.read_csv(os.path.join(output_directory, 'vgg19_history.csv'))
effb7_history_csv = pd.read_csv(os.path.join(output_directory, 'effb7_history.csv'))
#mv2_history_csv.rename(columns={ mv2_history_csv.columns[1]: "epoch" })
#mv2_history_csv.rename({'Unnamed: 0':'epoch'}, axis='columns')
fig = plt.figure(figsize = (20,20))
ax1 = fig.add_subplot(2, 3, 1) # row, column, position
ax2 = fig.add_subplot(2, 3, 2) # row, column, position
ax3 = fig.add_subplot(2, 3, 3) # row, column, position
ax4 = fig.add_subplot(2, 3, 4) # row, column, position
ax5 = fig.add_subplot(2, 3, 5) # row, column, position
ax6 = fig.add_subplot(2, 3, 6) # row, column, position

mv2_history_csv.rename( columns={'Unnamed: 0':'epoch'}, inplace=True)
vgg19_history_csv.rename( columns={'Unnamed: 0':'epoch'}, inplace=True)
effb7_history_csv.rename( columns={'Unnamed: 0':'epoch'}, inplace=True)
vgg19_history_csv.plot(x = "epoch", y=['loss','val_loss'], ax=ax1, title = 'Training vs Val loss for VGG19 model')
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss value")
ax1.set_ylim(min(vgg19_history_csv.loss.min(), vgg19_history_csv.val_loss.min(), mv2_history_csv.loss.min(), mv2_history_csv.val_loss.min(),effb7_history_csv.loss.min(), effb7_history_csv.val_loss.min()),max(vgg19_history_csv.loss.max(), vgg19_history_csv.val_loss.max(), mv2_history_csv.loss.max(), mv2_history_csv.val_loss.max(), effb7_history_csv.loss.max(), effb7_history_csv.val_loss.max()))

mv2_history_csv.plot(x = "epoch", y=['loss','val_loss'], ax=ax2, title = 'Training vs Val loss for MobileNetV2 model')
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss value")
ax2.set_ylim(min(vgg19_history_csv.loss.min(), vgg19_history_csv.val_loss.min(), mv2_history_csv.loss.min(), mv2_history_csv.val_loss.min(),effb7_history_csv.loss.min(), effb7_history_csv.val_loss.min()),max(vgg19_history_csv.loss.max(), vgg19_history_csv.val_loss.max(), mv2_history_csv.loss.max(), mv2_history_csv.val_loss.max(), effb7_history_csv.loss.max(), effb7_history_csv.val_loss.max()))

effb7_history_csv.plot(x = "epoch", y=['loss','val_loss'], ax=ax3, title = 'Training vs Val loss for EfficientNetB7 model')
ax3.set_xlabel("epoch")
ax3.set_ylabel("loss value")
ax3.set_ylim(min(vgg19_history_csv.loss.min(), vgg19_history_csv.val_loss.min(), mv2_history_csv.loss.min(), mv2_history_csv.val_loss.min(),effb7_history_csv.loss.min(), effb7_history_csv.val_loss.min()),max(vgg19_history_csv.loss.max(), vgg19_history_csv.val_loss.max(), mv2_history_csv.loss.max(), mv2_history_csv.val_loss.max(), effb7_history_csv.loss.max(), effb7_history_csv.val_loss.max()))

vgg19_history_csv.plot(x = "epoch", y=['accuracy','val_accuracy'], ax=ax4, title = 'Training vs Val accuracy for VGG19 model')
ax4.set_xlabel("epoch")
ax4.set_ylabel("accuracy value")
ax4.set_ylim(min(vgg19_history_csv.accuracy.min(), vgg19_history_csv.val_accuracy.min(), mv2_history_csv.accuracy.min(), mv2_history_csv.val_accuracy.min(), effb7_history_csv.accuracy.min(), effb7_history_csv.val_accuracy.min()),max(vgg19_history_csv.accuracy.max(), vgg19_history_csv.val_accuracy.max(), mv2_history_csv.accuracy.max(), mv2_history_csv.val_accuracy.max(), effb7_history_csv.accuracy.max(), effb7_history_csv.val_accuracy.max()))
mv2_history_csv.plot(x = "epoch", y=['accuracy','val_accuracy'], ax=ax5, title = 'Training vs Val accuracy for MobileNetV2 model')
ax5.set_xlabel("epoch")
ax5.set_ylabel("accuracy value")
ax5.set_ylim(min(vgg19_history_csv.accuracy.min(), vgg19_history_csv.val_accuracy.min(), mv2_history_csv.accuracy.min(), mv2_history_csv.val_accuracy.min(), effb7_history_csv.accuracy.min(), effb7_history_csv.val_accuracy.min()),max(vgg19_history_csv.accuracy.max(), vgg19_history_csv.val_accuracy.max(), mv2_history_csv.accuracy.max(), mv2_history_csv.val_accuracy.max(), effb7_history_csv.accuracy.max(), effb7_history_csv.val_accuracy.max()))
effb7_history_csv.plot(x = "epoch", y=['accuracy','val_accuracy'], ax=ax6, title = 'Training vs Val accuracy for EfficientNetb7 model')
ax6.set_xlabel("epoch")
ax6.set_ylabel("accuracy value")
ax6.set_ylim(min(vgg19_history_csv.accuracy.min(), vgg19_history_csv.val_accuracy.min(), mv2_history_csv.accuracy.min(), mv2_history_csv.val_accuracy.min(), effb7_history_csv.accuracy.min(), effb7_history_csv.val_accuracy.min()),max(vgg19_history_csv.accuracy.max(), vgg19_history_csv.val_accuracy.max(), mv2_history_csv.accuracy.max(), mv2_history_csv.val_accuracy.max(), effb7_history_csv.accuracy.max(), effb7_history_csv.val_accuracy.max()))
plt.savefig(args["accuracy_loss_plot"])
plt.show()

#############################################################################################

# loading the MobileNetV2 model back
mv2_model = keras.models.load_model(args["model2"])
vgg19_model = keras.models.load_model(args["model1"])
effb7_model = keras.models.load_model(args["model3"])

X_test = np.load(os.path.join(output_directory, "X_test.npy"))
X2_test = np.load(os.path.join(output_directory, "X2_test.npy"))
X3_test = np.load(os.path.join(output_directory, "X3_test.npy"))
y_test = np.load(os.path.join(output_directory, "y_test.npy"))
y2_test = np.load(os.path.join(output_directory, "y2_test.npy"))
y3_test = np.load(os.path.join(output_directory, "y3_test.npy"))

# Find the difference between the predicted values count between two models
def difference_predicted(model1, model2, X1, y1, X2, y2, model1_name, model2_name):
  print("[INFO] evaluating "+model1_name+" model predicted values...")
  m1 = model1.predict(X1, batch_size=64).round()
  # for each image in the testing set we need to find the index of the
  # label with corresponding largest predicted probability
  m1 = np.argmax(m1, axis=1)
  print("[INFO] evaluating "+model2_name+" model predicted values...")
  m2 = model2.predict(X2, batch_size=64).round()
  # for each image in the testing set we need to find the index of the
  # label with corresponding largest predicted probability
  m2 = np.argmax(m2, axis=1)
  print("Difference between the predicted values of " +model1_name+" model and "+model2_name+" model are: " + str(y1.shape[0]-np.count_nonzero(m1 == m2)) + " in "+ str(y1.shape[0]) + " records")

difference_predicted(mv2_model, vgg19_model, X_test, y_test, X2_test, y2_test, "MobilenetV2", "VGG19")
difference_predicted(mv2_model, effb7_model, X_test, y_test, X3_test, y3_test, "MobilenetV2", "Efficient Net")
difference_predicted(effb7_model, vgg19_model, X3_test, y3_test, X2_test, y2_test, "Efficient Net", "VGG19")
#############################################################################################
# Evaluate VGG19 model
print(vgg19_model.evaluate(X2_test, y2_test, batch_size = 64))

# Evaluate MobileNetV2 model
print(mv2_model.evaluate(X_test, y_test, batch_size = 64))

# Evaluate EFFB7 model
print(effb7_model.evaluate(X3_test, y3_test, batch_size = 64))
###########################################################################################
# Model Prediction for MobileNetV2
mv2_y_pred = mv2_model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score: ",accuracy_score(y_test, mv2_y_pred.round())*100)

from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report: \n", classification_report(mv2_y_pred.round(), y_test))

# Model Prediction for VGG19
vgg19_y_pred = vgg19_model.predict(X2_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score: ",accuracy_score(y2_test, vgg19_y_pred.round())*100)

from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report: \n", classification_report(vgg19_y_pred.round(), y2_test))

# Model Prediction for EFFB7 
effb7_y_pred = effb7_model.predict(X3_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score: ",accuracy_score(y3_test, effb7_y_pred.round())*100)

from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report: \n", classification_report(effb7_y_pred.round(), y3_test))
############################################################################################

# Confusion Matrix - {'No Mask': 0, 'Wrong Mask': 1, 'Mask': 2}
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
fig = plt.figure(figsize = (20,5))
ax1 = fig.add_subplot(1, 3, 1) # row, column, position
ax2 = fig.add_subplot(1, 3, 2) # row, column, position
ax3 = fig.add_subplot(1, 3, 3) # row, column, position
cm1 =confusion_matrix(y2_test.argmax(axis=1), vgg19_y_pred.round().argmax(axis=1))  
index = ['No Mask','Wrong Mask','Mask']  
columns = ['No Mask','Wrong Mask','Mask']  
cm_df1 = pd.DataFrame(cm1,columns,index) 
cm_df1.index.name = "vgg19_Actual"
cm_df1.columns.name = "vgg19_Predicted"
ax1.set_title("VGG19 Model")

cm2 =confusion_matrix(y_test.argmax(axis=1), mv2_y_pred.round().argmax(axis=1))  
index = ['No Mask','Wrong Mask','Mask']  
columns = ['No Mask','Wrong Mask','Mask']  
cm_df2 = pd.DataFrame(cm2,columns,index) 
cm_df2.index.name = "mv2_Actual"
cm_df2.columns.name = "mv2_Predicted"
ax2.set_title("MobileNetV2 Model")

cm3 =confusion_matrix(y3_test.argmax(axis=1), effb7_y_pred.round().argmax(axis=1))  
index = ['No Mask','Wrong Mask','Mask']  
columns = ['No Mask','Wrong Mask','Mask']  
cm_df3 = pd.DataFrame(cm3,columns,index) 
cm_df3.index.name = "effb7_Actual"
cm_df3.columns.name = "effb7_Predicted"
ax3.set_title("EfficientNetB7 Model")

sns.heatmap(cm_df1, annot=True, fmt = ".0f", ax=ax1)
sns.heatmap(cm_df2, annot=True, fmt = ".0f", ax=ax2)
sns.heatmap(cm_df3, annot=True, fmt = ".0f", ax=ax3)

plt.savefig(args["confusion_matrix"])
plt.show()
#######################################################################################
