import torch
import torch.nn as nn
import torchvision
import argparse
import numpy as np
from models import resnet50_cbam


def weights_init(m):
    classname =m.__class__.__name__
    if classname.find('Conv')!= -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!= -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

parser =argparse.ArgumentParser()
parser.add_argument('--batchsize',type=int,default=16,help='batch_size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--start_epoch',type=int,default=0,help='start epochs')
parser.add_argument('--epochs',type=int,default=100,help='total epochs')
parser.add_argument('--img_size',type=int,default=224)
parser.add_argument('--data_path',type =str,default ='..')

opt =parser.parse_args()
model =resnet50_cbam()
model.apply(weights_init)
model =model.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn =loss_fn.cuda()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((opt.img_size,opt.img_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    drop_last=True,
)



optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
for epoch in range(opt.epochs):
    losses=[]
    for i,(inputs,labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs,labels)
        loss.backward()
        losses.append(loss.item())       
        if i % 10 ==0:
            batch_mean_loss  = np.mean(losses)
            print('Training Batch[%d/%d]\t Class Loss: %.4f\t'           \
                            % ( i, len(dataloader) - 1, batch_mean_loss))
            losses.clear()