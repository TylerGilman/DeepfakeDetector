import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from RNN import RNN
from CNN import ConvNet
import cv2
import os

IMG_SIZE = 224

# Loads a video from the given path as a list of frames and resizes it to the given dimensions
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

#Constants
IMG_SIZE = 224
BATCH_SIZE = 1

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((270, 480)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def prepare_video(frames):
    #Am array to store the output of the CNN on each of the frames
    frame_features = np.zeros(shape = (1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype = 'float32')

    seq = 0
    for frame in frames:
        length = min(MAX_SEQ_LENGTH, frame.shape[0])
        for j in range(length):
            #Calculate the predicted class of that frame and add that to the temp array
            with torch.no_grad():
                frame_features[0, j, :] = modelcnn(transform(frame).unsqueeze(0))
        seq += 1
        if seq >= MAX_SEQ_LENGTH:
            break

        
    return frame_features

#Returns the CNN predicted result of the video based on each of its frames
def prepare_all_videos(root_dir):
    #Counting the number of videos and keeping a list of all of their file paths
    num_samples = len(os.listdir(root_dir + "/Fake")) + len(os.listdir(root_dir + "/Real"))
    video_paths = os.listdir(root_dir + "/Fake")
    video_paths += os.listdir(root_dir + "/Real")

    #Create an array that stores the class of its associated video as a one-hot vector
    #The index of the 1 is associated with where it would be in the output of the CNN
    labels = []

    for _ in range (len(os.listdir(root_dir + "/Fake"))):
        labels.append([0, 1])
        
    for _ in range (len(os.listdir(root_dir + "/Real"))):
        labels.append([1, 0])

    #Am array to store the output of the CNN on each of the frames
    frame_features = np.zeros(shape = (num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype = 'float32')

    #For each video
    for idx, path in enumerate(video_paths):
        #Gather all of its frames and add a batch dimenstion
        if labels[idx] == [0, 1]:
            frames = load_video(os.path.join(root_dir, "Fake", path))
        else:
            frames = load_video(os.path.join(root_dir, "Real", path))
        frames = frames[None,...]

        #Initialize placeholders to store features of the current video
        temp_frame_features = np.zeros(shape = (1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype = 'float32')

        seq = 0
        #For each frame in the video up o MAX_SEQ_LENGTH
        for frame in frames:
            length = min(MAX_SEQ_LENGTH, frame[1].shape[0])
            for j in range(length):
                #Calculate the predicted class of that frame and add that to the temp array
                with torch.no_grad():
                    temp_frame_features[0, j, :] = modelcnn(transform(frame[1]).unsqueeze(0))
                seq += 1
                if seq >= MAX_SEQ_LENGTH:
                    break
                
        #Squeeze the value of each frame into one value to represent the whole video
        frame_features[idx,] = temp_frame_features.squeeze()
        
        return frame_features, labels

#hyperparameters 
input_size = 2
sequence_length = 20
num_layers = 3
hidden_size = 3
num_classes = 2
learning_rate = 1e-4
batch_size = 64
num_epochs = 30 

cnnWeights = torch.load("./models/CNN.pth", map_location=torch.device('cpu'))
modelcnn = ConvNet()
modelcnn.load_state_dict(cnnWeights)


weights = torch.load("./models/RNN.pth", map_location=torch.device('cpu'))
modelrnn = RNN(input_size, hidden_size, num_layers, num_classes)
modelrnn.load_state_dict(weights)

video_path = "D:\Dataset\dataset\Train\Fake\\000_003.mp4"

video_data = load_video(video_path)
video = prepare_video(video_data)
print(video)

with torch.no_grad():
    prediction = modelrnn(video[0])
classes = ["Real", "Fake"]

probabilities = torch.nn.functional.softmax(prediction[0], dim = 0)

predicted_class = torch.argmax(probabilities).item()
certainty = probabilities[predicted_class].item()

print("Predicted class: {}, Certainty: {:.2%}".format(classes[predicted_class], certainty))