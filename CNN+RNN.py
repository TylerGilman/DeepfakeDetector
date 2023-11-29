import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16, resnet
from keras.layers import *
from keras.models import Model, Sequential
from keras import optimizers
from keras import regularizers

#Change this to be wherever your dataset is
dataset_path_string = "D:\Dataset\dataset\Train"
dataset_path_test_string = "D:\Dataset\dataset\Test"

dataset_path = os.listdir(dataset_path_string)
label_types = os.listdir(dataset_path_string)

#print(label_types) Prints ['fake, 'real'], the two labels

####################################
# Prepare training data
####################################
rooms = []

for item in dataset_path:
    # Get all the file names
    all_rooms = os.listdir(dataset_path_string + "/" + item)

    #Add them to the list
    for room in all_rooms:
        rooms.append((item, str('dataset/train' + '/' + item) + '/' + room))
    
#build a dataframe
train_df = pd.DataFrame(data = rooms, columns=['tag', 'video_name'])
# Prints a list of each video with it's label
#print(train_df.head())
#print(train_df.tail())

df = train_df.loc[:,['video_name', 'tag']]
df
df.to_csv('train.csv')

####################################
# Prepare test data
####################################
dataset_path = os.listdir(dataset_path_test_string)

room_types = os.listdir(dataset_path_test_string)

room = []

for item in dataset_path:
    #Get al the file names
    all_rooms = os.listdir(dataset_path_test_string + '/' + item)

    #add them to the list
    for room in all_rooms:
        rooms.append((item, str(dataset_path_test_string + '/' + item) + '/' + room))

#Build a dataframe
test_df = pd.DataFrame(data = rooms, columns= ['tag', 'video_name'])

df = test_df.loc[:,['video_name', 'tag']]
df 
df.to_csv('test.csv')

####################################
# Data Preperation
####################################
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

####################################
# Feed the videos to a network
####################################
IMG_SIZE = 224

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(x, y)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames = 0, resize = (IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

####################################
# Feature Extraction (CNN)
####################################

#My attempt at creating a custom feature extractor with PyTorch. Turned out to be incompatible
class ConvBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels) -> None:
        super().__init__()
        self.convos = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1),
                nn.ReLU()
            )
            for i in range(num_layers)]
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride = 2)

    def forward(self, x):
        for conv in self.convos:
            x = conv(x)
        x = self.downsample(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels, num_blocks, num_classes):
        super().__init__()
        first_channels = 64
        self.blocks = nn.ModuleList(
            [ConvBlock(
                2 if i==0 else 3,
                in_channels=(in_channels if i ==0 else first_channels*(2**(i-1))),
                out_channels=first_channels*(2**i))
            for i in range(num_blocks)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.cls = nn.Linear(first_channels*(2**(num_blocks - 1)), num_classes)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.cls(x)
        return x


#My attempt to make a custom feature extractor with Tensorflow
#This would be compatible if I could figure out how to train it
''' 
modelcnn = Sequential()
modelcnn.add(Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
modelcnn.add(Conv2D(16, (3,3), activation = 'relu',))
modelcnn.add(MaxPooling2D((3,3)))

modelcnn.add(Conv2D(32, (3, 3), activation='relu'))
modelcnn.add(Conv2D(32, (3,3), activation = 'relu'))
modelcnn.add(MaxPooling2D((2,2)))

modelcnn.add(Conv2D(64, (3, 3), activation='relu'))
modelcnn.add(Conv2D(64, (3,3), activation = 'relu'))
modelcnn.add(MaxPooling2D((2,2)))
modelcnn.add(Dropout(0.3))

modelcnn.add(Conv2D(32, (3,3), activation='relu'))
modelcnn.add(MaxPooling2D((2,2)))

modelcnn.add(Flatten())
modelcnn.add(Dense(512, activation='relu'))
modelcnn.add(Dropout(0.5))
modelcnn.add(Dense(1, activation='sigmoid'))

modelcnn.compile(loss='binary_crossentropy',
                 optimizer=optimizers.RMSprop(lr = 1e-4),
                 metrics = ['accuracy'])

modelcnn.fit_generator(train_data= , epochs = 5, validation_data=)
'''

#model = CNN(3, 4, 2) 
#feature_extractor = modelcnn
#create_feature_extractor(
#    model, return_nodes=['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3']
#)

#Instead, I'm using a pretrained feature extractor for noe
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights = "imagenet",
        include_top = False,
        pooling = "avg",
        input_shape = (IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")
print("created feature extractor")
feature_extractor = build_feature_extractor()

####################################
# Label Encoding
####################################

#Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

label_processor = keras.layers.StringLookup(num_oov_indices = 0, vocabulary=np.unique(train_df['tag']))
#print(label_processor.get_vocabulary()) prints ['fake', 'real']

labels = train_df['tag'].values
labels = label_processor(labels[..., None]).numpy()
#print(labels) prints a list if 0s and 1s representing 

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df['video_name'].values.tolist()

    #take all class Lables from train_df column named 'tag' and store in labels
    labels = df['tag'].values

    #convert class lables to label encoding
    labels = label_processor(labels[..., None]).numpy()

    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype='bool')
    frame_features = np.zeros(shape = (num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype = 'float32')

    #For each video
    for idx, path in enumerate(video_paths):
        #Gather all of its frames and add a batch dimenstion
        framse = load_video(os.path.join(root_dir, path))
        frames = framse[None,...]

        #Initialize placeholders to store the masks and features of the current video
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32'
        )

        #Extract features from the frames of the current video
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            temp_frame_mask[i, :length] = 1
        
        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    
    return (frame_features, frame_masks), labels

train_data, train_labels = prepare_all_videos(train_df, dataset_path_string)
test_data, test_labels = prepare_all_videos(test_df, dataset_path_test_string)

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

####################################
# RNN
####################################

def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')

    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask = mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation = 'relu')(x)
    output = keras.layers.Dense(len(class_vocab), activation = 'softmax')(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss = 'sparse_categprocal_crossentropy', optimizer = 'adam', metrics = ['accuracy']
    )
    return rnn_model

EPOCHS = 30

def run_experiment():
    filepath = './tmp/video_classifier'
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only = True, save_best_only = True, verbose = 1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs = EPOCHS,
        callbacks = [checkpoint]
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy {round(accuracy * 100, 2)}%")

    return history, seq_model
_, get_sequence_model = run_experiment()