from dataset import Dataset
from torchvision import transforms
import torch.optim as optim

from edar import EDAR

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from utils import AverageMeter
from tqdm import tqdm

# Define root path to high resolution DIV2k dataset
images_dir = '/Users/akhilbaweja/Desktop/Winter Assignments/805/project/data/DIV2K_valid_HR'
patch_size, jpeg_quality = 48, 40
transforms_train = transforms.Compose([transforms.ToTensor()])

# Load dataloader class
dataset = Dataset(images_dir, patch_size, jpeg_quality, transforms=transforms_train)

# Define available device for train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading model
model = EDAR().to(device)
dataloader = DataLoader(dataset=dataset,
                        batch_size=2,
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True)

# Define model paramters and loss function
lr = 1e-4
num_epochs = 100
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        #print(inputs.size(), labels.size())
        dummy_text_emb1 = torch.ones(1, 512, device=device)
        dummy_text_emb2 = torch.ones(1, 512, device=device)
        dummy_text_emb = torch.stack((dummy_text_emb1, dummy_text_emb2))

        optimizer.zero_grad()
        outs = model(inputs, dummy_text_emb)
        loss = criterion(outs, labels)
        print(f'Batch {i}: ', loss.item())
        loss.backward()
        optimizer.step()
