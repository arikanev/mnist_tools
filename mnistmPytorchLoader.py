import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler as lrs
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.utils.data.dataset
import pandas as pd
import os
from PIL import Image


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

mnistTrainSet = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
mnistTrainLoader = torch.utils.data.DataLoader(mnistTrainSet, batch_size=16,
                                          shuffle=True, num_workers=2)

mnistTestSet = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
mnistTestLoader = torch.utils.data.DataLoader(mnistTestSet, batch_size=16,
                                         shuffle=False, num_workers=2)


transformMnistm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


mnistmTrainSet = mnistmTrainingDataset(text_file ='Downloads/mnist_m/mnist_m_train_labels.txt',
                                       root_dir = 'Downloads/mnist_m/mnist_m_train')

mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet,batch_size=16,shuffle=True, num_workers=2)

mnistmTestSet = mnistmTestingDataset(text_file ='Downloads/mnist_m/mnist_m_test_labels.txt',
                                       root_dir = 'Downloads/mnist_m/mnist_m_test')

mnistmTestLoader = torch.utils.data.DataLoader(mnistmTestSet,batch_size=16,shuffle=True, num_workers=2)



class mnistmTestingDataset(torch.utils.data.Dataset):
    
    def __init__(self,text_file,root_dir,transform=transformMnistm):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all test images
        """
        self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))
        self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.name_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.label_frame.iloc[idx, 0]
        #labels = labels.reshape(-1, 2)
        sample = {'image': image, 'labels': labels}
        
        return sample



    
class mnistmTrainingDataset(torch.utils.data.Dataset):
    
    def __init__(self,text_file,root_dir,transform=transformMnistm):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))
        self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.name_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.label_frame.iloc[idx, 0]
        #labels = labels.reshape(-1, 2)
        sample = {'image': image, 'labels': labels}
        
        return sample
        
        
        
#test trainloader ouput + batchsize (mnist-m)     
for i_batch,sample_batched in enumerate(mnistmTrainLoader,0):
    print("training sample for mnist-m")
    print(i_batch,sample_batched['image'],sample_batched['labels'])
    if i_batch == 0:
        break
        
#test testloader output + batchsize (mnist-m)   
for i_batch,sample_batched in enumerate(mnistmTestLoader,0):
    print("testing sample for mnist-m")
    print(i_batch,sample_batched['image'],sample_batched['labels'])
    if i_batch == 0:
       break     
        
#test trainloader output+batchsize (mnist)   
for i_batch,sample_batched in enumerate(mnistTrainLoader,0):
    print("training sample for mnist")
    print(i_batch,sample_batched[0],sample_batched[1])
    if i_batch == 0:
        break
#test testloader output+batchsize (mnist)
for i_batch,sample_batched in enumerate(mnistTestLoader,0):
    print("testing sample for mnist")
    print(i_batch,sample_batched[0],sample_batched[1])
    if i_batch == 0:
        break
