import torch
import argparse
from models import resnet50_cbam
parser =argparse.ArgumentParser()
parser.add_argument('--batchsize',type=int,default=64,help='batch_size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--start_epoch',type=int,default=0,help='start epochs')
parser.add_argument('--epochs',type=int,default=100,help='total epochs')
opt =parser.parse_args()
model =resnet50_cbam()
model =model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4) 