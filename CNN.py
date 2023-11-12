import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

#Hyper parameters
#Set to arbitrary values noe, to be tuned later
num_epochs = 4
batch_size = 4
learning_rate = .001

#######################
# Pre-processing and loading data
#######################

#Some transform to normalize the data
#Again, completely arbitrary. I don't know if this will work 
#with the dataset that we are using
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

#Load in the datasets
#Replace Nonw with the actual datasets
train_dataset = None
test_dataset = None
train_loader = None
test_loader = None

classes = {"Real", "Fake"}

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #Sample of what the neural network could look like.
        #Will likely need more layers to detect deep fakes
        self.conv1 = nn.Conv2d('''Input channel sizes here (input, output, kernel)''')
        self.pool = nn.MaxPool2d('''Input kernel size and stride here''')
        self.conv2 = nn.Conv2d('''input channel should be same as output channel above''')
        self.fc1 = nn.Linear('''Input input and output sizes here''')
        self.fc1 = nn.Linear('''Input input and output sizes''')
        self.fc3 = nn.Linear('''Input input and output sizes. Output should be 2 because this is the final layer''')

    def forward(self, x):
        #Convolutional part
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        #Flatten
        x = x.view(-1, '''Input diension here''')
        
        #Linear part
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = ConvNet()

#Loss function and optmizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Training the network
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#Evaluating the network
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    acc = 100 * n_correct / n_samples
    print(f"Accuracy of the networdL {acc} %")

    for i in range(2):
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]: {acc}} %")

#Save the neural network so we only have to train it once
torch.save(model.state_dict(), '''Path to save''')