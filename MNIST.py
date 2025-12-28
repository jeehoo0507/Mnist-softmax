import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

UseCuda = torch.cuda.is_available()
device = torch.device("cuda" if UseCuda else "cpu")

print("device:", device)

random.seed(0)
torch.manual_seed(0)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(0)

training_epochs = 15
batch_size = 100

class MnistDataset(Dataset):
    def __init__(self, train=True):
        self.data = dsets.MNIST(root='MNIST_data/',
                                train=train,
                                transform=transforms.ToTensor(),
                                download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

data_set = MnistDataset(train=True)
data_loader = DataLoader(dataset=data_set, batch_size=batch_size,
                         shuffle=True, drop_last=True)

class MnistLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10, bias=True)

    def forward(self, x):
        return self.linear(x)

model = MnistLinear().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for x, y in data_loader:
        x = x.view(-1, 28*28).to(device)
        y = y.to(device)

        hypothesis = model(x)
        cost = criterion(hypothesis, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('훈련 끝')


with torch.no_grad():
    test_dataset = MnistDataset(train=False)
    
    X_test = test_dataset.data.data.view(-1, 28 * 28).float().to(device)
    Y_test = test_dataset.data.targets.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    r = random.randint(0, len(test_dataset) - 1)
    X_single_data = test_dataset.data.data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = test_dataset.data.targets[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(test_dataset.data.data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()