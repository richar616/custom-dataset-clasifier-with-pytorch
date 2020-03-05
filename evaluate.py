
import torch
from torchvision import transforms
import os 
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from torch import nn
from torch.nn import functional as F
import time

class Loader(Dataset):
    def __init__(self, split, trasform = None):
        base_dir='./datasets/'

        path =os.path.join(base_dir,'64x64_SIGNS/{}'.format(split))

        files = os.listdir(path)

        self.filenames = [os.path.join(path,f) for f in files if f.endswith('.jpg')]

        self.targets = [int(f[0]) for f in files]

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])

        if self.transform:
            image = self.transform(image)
        
        return image, self.targets[idx]



transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=9,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=9,out_channels=12,kernel_size=5)
        self.fc1 = nn.Linear(2028,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,6)
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        x = self.pool(F.relu( self.conv1(x)))
        x = self.pool(F.relu( self.conv2(x)))

        x = x.view(-1,2028)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.log_softmax(x,dim=1)
        return x
net = Net()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
testSet = Loader('test_signs',transform)
start = time.time()
with torch.no_grad():
    mistakes = 0
    for i in range(100):
        net.load_state_dict(torch.load('model.pth'))
        image, label = testSet[i]
        image = torch.tensor(image, dtype= torch.float32, device=device).unsqueeze(0)
        out = net(image)
        porcent = torch.nn.functional.softmax(out, dim=1)[0]*100
        _, index = torch.max(out, dim=1)
        print('input: ',label,'    output: ',index.item(),'   acuracy: ',porcent[index[0]].item())
        if label != index.item():
            mistakes +=1
    stop = time.time()
    Etime = start-stop
    print('wrong predictions: ',mistakes,'        execution time: ',Etime)








