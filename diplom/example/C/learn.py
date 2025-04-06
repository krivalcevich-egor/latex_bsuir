import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Working on device {device}")


# Transform image to 1x784 and normalize colors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download dataset
trainset_full = datasets.MNIST('../../../Datasets', download=False, train=True, transform=transform)
testset = datasets.MNIST('../../../Datasets', download=False, train=False, transform=transform)

val_size, train_size = 1000, 59000
val_dataset, train_dataset = torch.utils.data.random_split(trainset_full, [val_size, train_size], generator=torch.Generator().manual_seed(27))

# Create dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False) 


class L2DST(nn.Module):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        out = torch.nn.functional.tanh(self.dense1(X))
        out1 = torch.transpose(out, -1, -2)
        out = torch.nn.functional.tanh(self.dense2(out1))
        out = torch.transpose(out, -2, -1)
        return out

    def get_embeddings(self,X):
        out = torch.nn.functional.tanh(self.dense1(X))
        e1 = out
        out = torch.transpose(e1, -1, -2)
        out = torch.nn.functional.tanh(self.dense2(out))
        e2 = out
        out = torch.transpose(e2, -2, -1)
        
        return e1,e2
    
class L2DST_1l(nn.Module):
    def __init__(self, input_size, num_classes, device='cpu'):
        super(L2DST_1l, self).__init__()
        
        self.L2DST = L2DST(input_size,input_size)
        
        dropout = 0.05
        self.dropout = nn.Dropout(dropout)
        
        # self.BN1 = nn.LayerNorm((1,input_size,input_size))
        
        self.W_o = nn.Linear(input_size*input_size, num_classes, device=device)
        
        self.num_classes = num_classes

        self.log_softmax = nn.LogSoftmax(dim=1)
        
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, x):
        # Multiply input by weights and add biases
        
        out = self.L2DST(x)
        
        out = out.reshape(out.shape[0],-1)
        
        out = self.log_softmax(self.dropout(self.W_o(out)))
        
        return out
    
    def get_embeddings(self,x):
        e1,e2 = self.L2DST.get_embeddings(x)
        
        return e1,e2
    
    
    # Build the Neural Network
input_size = 28  # 28x28 images flattened
output_size = 10  # 10 classes for digits 0-9

N_epoch = 300
model = L2DST_1l(input_size,output_size, device=device)
model.load_state_dict(torch.load(f'model_backup\L2DST_1l_epoch_{N_epoch}.pth', map_location=torch.device('cpu'))) #
# print(model)
model.to(device)

# criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss which includes softmax
criterion = nn.NLLLoss()  # Use CrossEntropyLoss which includes softmax
optimizer = optim.Adam(model.parameters(), lr=15e-4, weight_decay=0e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97, verbose=False)

# Track loss
loss_list = []
val_loss_list = []

# Training the network
epochs = 70
time0 = time()

for epoch in range(epochs):
    running_loss = 0

    # for images, labels in tqdm(trainloader, leave=False):
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        images = images.squeeze(1)                 

        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)               
        
        # This is where the model learns by back propagating
        loss.backward()
        
        # And optimizes its weights here
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
    
    # validation
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.squeeze(1)                 
            
            output = model(images)
            val_loss = criterion(output, labels)
    val_loss = val_loss / len(val_loader)
    val_loss_list.append(val_loss)
        
    CE_curr = running_loss / len(trainloader)
    loss_list.append(CE_curr)
    # if (epoch%10)==0:
    print(f"Epoch {epoch} - Training loss: {CE_curr:.5f}, Val loss: {val_loss:.5f}")
    
    
print(f"\nTraining Time (in minutes) = {(time()-time0)/60:.2f}")

# Convert lists to numpy arrays
loss_array = np.array(loss_list)

val_loss = [val.cpu().numpy() for val in val_loss_list]

val_loss_array = np.array(val_loss)

# Save the model
torch.save(model.state_dict(), f'model_backup\L2DST_1l_epoch_{N_epoch+epochs}.pth')

# Plot NLL_loss
plt.figure(figsize=(5, 2))
plt.plot(range(len(loss_array))[1:], loss_array[1:], label='Train')
plt.plot(range(len(val_loss_array)), val_loss_array, label='Validation')

plt.xlabel('Epochs')
plt.ylabel('NNLoss')
plt.title('NNLoss')
plt.legend()
plt.show()