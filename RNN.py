import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

#Very simple RNN. Can me expanded
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first='''Insert proper bool here, depends on dataset''')

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