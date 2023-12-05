import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from CNN import ConvNet
import cv2
import os
import matplotlib.pyplot as plt

#Constants
IMG_SIZE = 224
BATCH_SIZE = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2

if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device=torch.device("mps")
    elif torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    #The file paths of where the video data is being store
    #should be seperated in two folders, "Fake" and "Real"
    train_videos_path = "D:\Dataset\dataset\Train"
    test_videos_path = "D:\Dataset\dataset\Test"

    # Loading in the pretrained CNN. Used to train the RNN
    feature_extractor = torch.load("./models/CNN.pth", map_location=torch.device('cpu'))
    model = ConvNet()
    model.load_state_dict(feature_extractor)

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

    # Crops out the center square of the given frame
    def crop_center_square(frame):
        y, x = frame.shape[0:2]
        min_dim = min(x, y)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


    # Transforms to apply to each frame when passed into the model 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((270, 480)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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

            #For each frame in the video up o MAX_SEQ_LENGTH
            for i, frame in enumerate(frames):
                length = min(MAX_SEQ_LENGTH, frame.shape[0])
                for j in range(length):
                #Calculate the predicted class of that frame and add that to the temp array
                    with torch.no_grad():
                        temp_frame_features[i, j, :] = model(transform(frame[None, j, :][0]).unsqueeze(0))
                
            #Squeeze the value of each frame into one value to represent the whole video
            frame_features[idx,] = temp_frame_features.squeeze()

            print(idx)
        
        return frame_features, labels

    #Loading in the data
    train_data, train_labels = prepare_all_videos(train_videos_path)
    test_data, test_labels = prepare_all_videos(test_videos_path)

#hyperparameters 
input_size = 2
sequence_length = 20
num_layers = 20
hidden_size = 3
num_classes = 2
learning_rate = 1e-4
batch_size = BATCH_SIZE
num_epochs = 100

#Simple RNN class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size * sequence_length, out_features=num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        #Forward prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Initialize the RNN
    modelrnn = RNN(input_size, hidden_size, num_layers, num_classes)
    modelrnn.to(device)

    #Formatting the training data as tensors
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)

    #Formatting the testing data as tensors
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle = False, pin_memory=True)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelrnn.parameters(), lr = learning_rate)
    
    class_labels = ["Real", "Fake"]

    loss_values = []
    
    #Train network
    for epoch in range(num_epochs):
        iter = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)

            #forward
            scores = modelrnn(data)
            labels_in_batch = train_labels_tensor[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + scores.shape[0]]
            loss = criterion(scores, labels_in_batch)

            loss_values.append(loss.item())

            #backward
            optimizer.zero_grad()
            loss.backward()

            #optimizer step
            optimizer.step()
            iter += 1

    #Check accuracy
    num_correct = 0
    num_samples = 0
    modelrnn.eval()
    
    #Save the neural network so we only have to train it once
    torch.save(modelrnn.state_dict(), './models/RNN.pth')

    with torch.no_grad():
        done = False
        n_class_correct = [0 for i in range(2)]
        n_class_samples = [0 for i in range(2)]
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            
            scores = modelrnn(data)

            _, predictions = torch.max(scores, 1)
            _, classes = test_labels_tensor[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + scores.shape[0]].max(1)

            num_correct += (predictions == classes).sum()
            num_samples += predictions.size(-1)

            for i in range(classes.size(0)):
                label = classes[i]
                pred = predictions[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

            acc = 100 * num_correct / num_samples

            if done:
                break

        print(f"Accuracy of the network {acc} %")

        for i in range(2):
            acc = 100 * n_class_correct[i] / n_class_samples[i]
            print(f"Accuracy of {class_labels[i]}: {acc:.2f} %")

    #Print the loss function over time
    plt.plot(range(len(loss_values)), loss_values)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.savefig("RNN Loss over interations")

    #Save the neural network so we only have to train it once
    torch.save(modelrnn.state_dict(), './models/RNN.pth')
