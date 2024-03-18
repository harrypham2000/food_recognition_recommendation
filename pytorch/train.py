import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch import optim 
import matplotlib.pyplot as plt

# Set the device to GPU if available 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Choose the EfficientNet model
model = EfficientNet.from_pretrained

# Preprocessing step with augmentation for training data 
pre_process = torchvision.datasets.Food101(root='./data', download=True, split='train', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.ColorJitter(),
]))
# Data loader 
data_loader = torch.utils.data.DataLoader(pre_process, batch_size=32, shuffle=True)

#Classification layer
num_feature=model.modules['_fc'].in_features
num_class=101
model._fc=torch.nn.Linear(num_feature,num_class)
model=model.to(device)
#Loss function and optimizer
loss_func=torch.nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)

#Training 

num_epochs=50
for epoch in range(num_epochs):
    epoch_numbers = []  # Define epoch_numbers
    loss_values = []  # Define loss_values

    model.train()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        # Append epoch number and loss value
        epoch_numbers.append(epoch+1)
        loss_values.append(loss.item())

    # Plot the loss values
    plt.plot(epoch_numbers, loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
# Save the model
torch.save(model.state_dict(), 'model.pth')