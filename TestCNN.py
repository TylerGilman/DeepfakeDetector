import torch
import torchvision.transforms as transforms
from PIL import Image
from CNN import ConvNet

# Initialize your model
model = ConvNet()

# Load the saved state_dict
print("Model architecture before loading state_dict:")
print(model)

# Load the saved state_dict
model.load_state_dict(torch.load('./models/CNN.pth', map_location=torch.device('cuda')))

print("\nModel architecture after loading state_dict:")
print(model)

model.eval()
transform = transforms.Compose([
    transforms.Resize((270, 480)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Load and preprocess the input image
image_path = 'data/split/train/original_images/21.jpg'
image = Image.open(image_path)
input_image = transform(image).unsqueeze(0)

# Make predictions
with torch.no_grad():
    output = model(input_image)
classes = ["Real", "Fake"]

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the predicted class and its associated probability
predicted_class = torch.argmax(probabilities).item()
certainty = probabilities[predicted_class].item()

print("Predicted class: {}, Certainty: {:.2%}".format(classes[predicted_class], certainty))
