import os 
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import torch
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


from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


trainSet = Loader('train_signs',transform)
trainLoader = DataLoader(trainSet, batch_size=2)

valSet = Loader('val_signs',transform)
valLoader = DataLoader(valSet, batch_size=1)

testSet = Loader('test_signs',transform)
testLoader = DataLoader(testSet, batch_size=35)

dataLoader = {'train':trainLoader,'val':valLoader,'test':testLoader}


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


device = ('cuda' if torch.cuda.is_available() else 'cpu')


net = Net()
net.to(device)


from torch import optim 

loss_fn = nn.NLLLoss()

optimizer = optim.Adam(net.parameters(),lr=0.001)


epocs = 1

start = time.time()
for epoch in range(epocs):
    

    for inputs, targets in dataLoader['train']:

        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()#after each epoc i going to research for a new best result so i put the gradient on 0 cause i'll start all over again

        outputs = net(inputs)#fitting my images in the network to obtain a result to be used to predict 
        

        loss = loss_fn(outputs, targets) #get the loss

        loss.backward() #calculate the gradients automatically
 
        optimizer.step() #upgrade parametters

        print('loss: ',loss.item(),'      epoc: ',epoch)

stop = time.time()
Etime = start-stop 
print('execution time: ',Etime)
torch.save(net.state_dict(),'model.pth')




        




