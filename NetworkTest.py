import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from RNN import RNN
from CNN import ConvNet
import cv2
from PIL import Image
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

    frames = frames[None,...]


    for i, frame in enumerate(frames):
        length = min(MAX_SEQ_LENGTH, frame.shape[0])
        for j in range(length):
            #Calculate the predicted class of that frame and add that to the temp array
            with torch.no_grad():
                frame_features[i, j, :] = modelcnn(transform(frame[None, j, :][0]).unsqueeze(0))
    
    return frame_features

#hyperparameters 
input_size = 2
sequence_length = 20
num_layers = 3
hidden_size = 3
num_classes = 2

cnnWeights = torch.load("./models/CNN.pth", map_location=torch.device('cpu'))
modelcnn = ConvNet()
modelcnn.load_state_dict(cnnWeights)
modelcnn.eval()


weights = torch.load("./models/RNN.pth", map_location=torch.device('cpu'))
modelrnn = RNN(input_size, hidden_size, num_layers, num_classes)
modelrnn.load_state_dict(weights)
modelrnn.eval()

running = True

while running:

    path = input("Input path to video or image to classify \n")
    
    video = False
    image = False

    if path.endswith(".mp4"):
        video = True
    elif path.endswith(".jpg"):
        image = True
    elif path.equals("stop"):
        break

    if video:
        video_data = load_video(path)
        video = prepare_video(video_data)
        video = torch.tensor(video)


        with torch.no_grad():
            prediction = modelrnn(video)
        classes = ["Real", "Fake"]

        probabilities = torch.nn.functional.softmax(prediction[0], dim = 0)

        predicted_class = torch.argmax(probabilities).item()
        certainty = probabilities[predicted_class].item()

        print("Predicted class: {}, Certainty: {:.2%}".format(classes[predicted_class], certainty))
    elif image:
        transform = transforms.Compose([
            transforms.Resize((270, 480)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Load and preprocess the input image

        image = Image.open(path)
        input_image = transform(image).unsqueeze(0)
        # Make predictions
        with torch.no_grad():
            output = modelcnn(input_image)
        classes = ["Real", "Fake"]

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the predicted class and its associated probability
        predicted_class = torch.argmax(probabilities).item()
        certainty = probabilities[predicted_class].item()

        print("Predicted class: {}, Certainty: {:.2%}".format(classes[predicted_class], certainty))
    else:
        print("Invalid path entered")