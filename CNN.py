import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision
import numpy as np
import os
import random
from shutil import copyfile

if __name__ == "__main__":
    # Create training and testing datasets from 2 image folders
    TOTAL_IMAGE_NUMBER = 1000
    def split_data(input_folder, output_folder, split_ratio=0.8):
        # Create output folders for training and testing data
        train_folder = os.path.join(output_folder, 'train')
        test_folder = os.path.join(output_folder, 'test')

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # Iterate through each class folder
        for class_folder in os.listdir(input_folder):
            class_path = os.path.join(input_folder, class_folder)
            images = os.listdir(class_path)
            random.shuffle(images)

            # Split the images into training and testing sets
            split_index = int(split_ratio * len(images))
            train_images = images[:split_index]
            test_images = images[split_index:]

            # Copy images to the corresponding folders
            for image in train_images:
                src_path = os.path.join(class_path, image)
                dest_path = os.path.join(train_folder, class_folder, image)
                os.makedirs(os.path.join(train_folder, class_folder), exist_ok=True)
                copyfile(src_path, dest_path)

            for image in test_images:
                src_path = os.path.join(class_path, image)
                dest_path = os.path.join(test_folder, class_folder, image)
                os.makedirs(os.path.join(test_folder, class_folder), exist_ok=True)
                copyfile(src_path, dest_path)

    # Example usage
    split_data('data/full_dataset_images', 'data/split', split_ratio=0.8)

    #Hyper parameters
    #Set to arbitrary values noe, to be tuned later
    num_epochs = 1
    batch_size = 16
    learning_rate = .001

    #######################
    # Pre-processing and loading data
    #######################

    #Some transform to normalize the data
    #Again, completely arbitrary. I don't know if this will work 
    #with the dataset that we are using

    # Load and preprocess images
    transform = transforms.Compose([
        transforms.Resize((270, 480)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Load in the datasets
    #Replace None with the actual datasets
    train_dataset = torchvision.datasets.ImageFolder(root='./data/split/train/', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root='./data/split/test', transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


    classes = ["Real", "Fake"]

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Sample of what the neural network could look like.
        # Will likely need more layers to detect deep fakes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(67 * 120 * 16, 64)
        self.fc2 = nn.Linear(64, 2)  # Adjust the input size

    def forward(self, x):
        # Convolutional part
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 67 * 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

if __name__ == "__main__":
  import os
  import gc


  torch.cuda.empty_cache()
  gc.collect()
  device = torch.device("cuda")
  model = ConvNet().to(device)

  # Ensure that model weights are on the GPU
  model = model.to(device)

  #Loss function and optmizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


  #Training the network
  n_total_steps = len(train_loader)
  gc.collect()

  for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)  # Move data to GPU

      print("Input data device:", images.device)

      #forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      #backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      torch.cuda.empty_cache()
    gc.collect()



  #Evaluating the network
  with torch.no_grad():
      n_correct = 0
      n_samples = 0
      n_class_correct = [0 for i in range(2)]
      n_class_samples = [0 for i in range(2)]
      for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)  # Move data to GPU

          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          n_samples += labels.size(0)

          for i in range(len(classes)):
              label = labels[i]
              pred = predicted[i]
              if (label == pred):
                  n_class_correct[label] += 1
                  n_correct += 1
              n_class_samples[label] += 1
          model_accuracy = 100.0 * n_correct / n_samples

      for i in range(2):
          acc = 100 * n_class_correct[i] / n_class_samples[i]
          print(f"Accuracy of {classes[i]}: {acc:.2f} %")



  #Save the neural network so we only have to train it once
  torch.save(model.state_dict(), './models/CNN.pth')