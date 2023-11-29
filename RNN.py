import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from CNN import ConvNet
import cv2
import keras
import pandas as pd
import os
from torchvision.datasets import ImageFolder

#hyperparameters 
input_size = None
sequence_length = None
num_layers = None
hidden_size = None
num_classes = None
learning_rate = None
batch_size = None
num_epochs = None

#Load in the datasets
#Replace None with the actual datasets
train_dataset = None
test_dataset = None
train_loader = None
test_loader = None

########################################
# Prepare Video Data
########################################

# Change this to be wherever your dataset is the videos
# Used to train an RNN
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

feature_extractor = torch.load("./models/CNN.pth", map_location=torch.device('cpu'))
model = ConvNet()
model.load_state_dict(feature_extractor)

label_processor = keras.layers.StringLookup(num_oov_indices = 0, vocabulary=np.unique(train_df['tag']))

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100 

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

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


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(x, y)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


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
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None,...]

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
                temp_frame_features[i, j, :] = model(batch[None, j, :])
            temp_frame_mask[i, :length] = 1
        
        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    
    return (frame_features, frame_masks), labels

train_data, train_labels = prepare_all_videos(train_df, dataset_path_string)
test_data, test_labels = prepare_all_videos(test_df, dataset_path_test_string)


#Very simple RNN. Can me expanded
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=Insert proper bool here, depends on dataset)

        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zoros(self.num_layers, x.size(0), self.hidden_size)

        #Forward prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #optimizer step
        optim.step()

#Check accuracy
num_correct = 0
num_samples = 0
model.eval()

while torch.no_grad():
    for x, y in test_loader:
        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)
    acc = 100 * num_correct / num_samples
    print(f"Accuracy of the networdL {acc} %")
