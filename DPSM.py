# This is a pytoch implementation of DISTS metric, which is adapted for building texture comparison. For original code, please refer to: https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/DISTS_pt.py
# Please refer to their paper: "Image quality assessment: Unifying structure and texture similarity" (DOI: 10.1109/TPAMI.2020.3045810)
# Requirements: python >= 3.6, pytorch >= 1.0
# Testing command1: python -m torch.distributed.launch --nproc_per_node=4 DPSM.py --imgpairfile /media/aaron/E/Sample.csv



import numpy as np
import os, sys
from PIL import Image
import pandas as pd
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tqdm import tqdm
import logging



class StreetscapeDataset(Dataset): 
    def __init__(self, imgpairfile, imgdir, maskdir, istest, resize=False):
        #"imgpairfile" is a file path and the file contains names of image pair, associate mask and classification of change. Note: "imgpairfile" must contain 'test' if it is for testing. Otherwise, it's for training. 
        #"istest" indicates whether is a test dataset or training dataset.
        
        attr = pd.read_csv(imgpairfile)
        self.indlist = list(attr.loc[:,'ID'])
        self.imglist1 = list(imgdir + "/" + attr.loc[:,'img1'] + ".png")
        self.imglist2 = list(imgdir + "/" + attr.loc[:,'img2'] + ".png")
        self.masklist = list(maskdir + "/" + attr.loc[:,'mask'] + ".png")
        self.targetlist = [-1] * len(self.indlist) if istest else list(((attr.loc[:,'renovation']+1)/2).astype(int)) #Convert from [-1,1] to [0,1]
        self.resize = resize
                    
    def __len__(self):
        return len(self.indlist)

    def __getitem__(self, idx):
        #"image" has to be an object from PIL images, rather than a tensor.
        img1 = Image.open(self.imglist1[idx]).convert("RGB")
        img2 = Image.open(self.imglist2[idx]).convert("RGB")
        mask = Image.open(self.masklist[idx]).convert("L")
        
        
        if self.resize and min(img1.size)>256:
            img1 = transforms.functional.resize(img1,256)
        if self.resize and min(img2.size)>256:
            img2 = transforms.functional.resize(img2,256)
        if self.resize and min(mask.size)>256:
            mask = transforms.functional.resize(mask,256)
        
        
        
        img1 = transforms.ToTensor()(img1) 
        img2 = transforms.ToTensor()(img2) 
        mask = torch.as_tensor(np.array(mask), dtype=torch.bool)
        mask = mask.unsqueeze(0)
        
        return (self.indlist[idx], img1, img2, mask, self.targetlist[idx])

        
        
class L2pooling(nn.Module): 
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels        
        a = np.hanning(filter_size)[1:-1] 
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g) 
        
        
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))  

    def forward(self, input):
        input = input**2 
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()
        

class DISTS(torch.nn.Module):
    def __init__(self): 
        super(DISTS, self).__init__()
        """
        All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. More details about torchvision.models at: https://pytorch.org/vision/0.10/models.html
        A detailed explanation of vgg16 structure is discussed here: https://blog.csdn.net/Geek_of_CSDN/article/details/84343971. Also the paper: https://arxiv.org/pdf/1409.1556.pdf
        """
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        
        
        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential()
        self.stage5 = nn.Sequential()       
        
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x]) 
        self.stage2.add_module(str(4), L2pooling(channels=64)) 
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False
        
        
        self.fc1 = nn.Linear(2950, 1000) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(1000, 500)  
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(500, 250)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout()
        self.fc4 = nn.Linear(250, 100)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout()
        self.fc5 = nn.Linear(100, 50)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout()
        self.fc6 = nn.Linear(50, 10)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout()
        self.fc7 = nn.Linear(10, 2)
        
        

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)) 
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]      
        
        
            
        
    def shrinkMask(self, mask, shrinkfactors = (0.5, 0.5)): 
        h_factor, w_factor = shrinkfactors
        maskh = round(mask.size()[2] * h_factor)
        maskw = round(mask.size()[3] * w_factor)
        posb, posc, posh, posw = (mask!=0).nonzero(as_tuple=True)
        posh = torch.round(posh * h_factor).int()
        posh[posh >= maskh] = maskh - 1
        posw = torch.round(posw * w_factor).int()
        posw[posw >= maskw] = maskw - 1
        pos = [(a.item(), b.item(), c.item(), d.item()) for a, b, c, d in zip(posb, posc, posh, posw)]
        pos = list(set(pos)) 
        posb, posc, posh, posw = zip(*pos)
        mask = torch.zeros(mask.size()[0], 1, maskh, maskw, dtype = torch.int8)
        mask[posb, posc, posh, posw] = 1
        return mask.type(torch.bool)
  
        
        
    def forward_once(self, x, maskx):         
        featx = x * maskx 
        h = (x-self.mean)/self.std
        h = self.stage1(h)        
        h_relu1_2 = h * maskx
        
        h = self.stage2(h)
        maskx1 = self.shrinkMask(maskx).to(h.device)
        h_relu2_2 = h * maskx1 
        
        h = self.stage3(h)        
        maskx2 = self.shrinkMask(maskx1).to(h.device)
        h_relu3_3 = h * maskx2
        
        h = self.stage4(h)        
        maskx3 = self.shrinkMask(maskx2).to(h.device)
        h_relu4_3 = h * maskx3
        
        h = self.stage5(h)
        maskx4 = self.shrinkMask(maskx3).to(h.device)
        h_relu5_3 = h * maskx4
        
        return [(featx, maskx), (h_relu1_2, maskx), (h_relu2_2, maskx1), (h_relu3_3, maskx2), (h_relu4_3, maskx3), (h_relu5_3, maskx4)]

    def forward(self, x, y, mask, target, istest, require_grad=False): 
        if require_grad: 
            res0 = self.forward_once(x, mask)
            res1 = self.forward_once(y, mask)            
            feats0, mask0 = map(list,zip(*res0))
            feats1, mask1 = map(list,zip(*res1))  
        else:
            with torch.no_grad():
                res0 = self.forward_once(x, mask)
                res1 = self.forward_once(y, mask)            
                feats0, mask0 = map(list,zip(*res0))
                feats1, mask1 = map(list,zip(*res1))  
        dist1 = torch.empty(0, device=x.device)
        dist2 = torch.empty(0, device=x.device) 
        c1 = 1e-6
        c2 = 1e-6
        
        for k in range(len(self.chns)):            
            area0 = mask0[k].sum([2,3], keepdim=True)
            x_mean = feats0[k].sum([2,3], keepdim=True) / area0
            y_mean = feats1[k].sum([2,3], keepdim=True) / area0
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = torch.cat((dist1, S1.squeeze()), 1)

            x_var = ((feats0[k]-x_mean * mask0[k])**2).sum([2,3], keepdim=True) / area0
            y_var = ((feats1[k]-y_mean * mask0[k])**2).sum([2,3], keepdim=True) / area0
            xy_cov = (feats0[k]*feats1[k]).sum([2,3], keepdim=True) / area0 - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = torch.cat((dist2, S2.squeeze()), 1)

        
        
        dist12 = torch.cat((dist1, dist2), 1)
           
        score = self.fc1(dist12)
        score = self.relu1(score)
        score = self.dropout1(score)
        score = self.fc2(score)
        score = self.relu2(score)
        score = self.dropout2(score)
        score = self.fc3(score)
        score = self.relu3(score)
        
        score = self.dropout3(score)        
        score = self.fc4(score)
        score = self.relu4(score)
        score = self.dropout4(score)
        score = self.fc5(score)
        score = self.relu5(score)
        score = self.dropout5(score)
        score = self.fc6(score)
        score = self.relu6(score)
        score = self.dropout6(score)
        score = self.fc7(score)
        
        
        if istest:
            loss = None
        else:            
            celoss = nn.CrossEntropyLoss()        
            loss = celoss(score, target)            
             
        return score


    
            
            

if __name__ == '__main__':    
   
    import argparse
    parser = argparse.ArgumentParser(description='DPSM training or testing') # To create an ArgumentParser object.
    parser.add_argument('--imgpairfile', type=str, default='/media/aaron/E/Trajectories.csv')
    parser.add_argument('--imgdir', type=str, default='/media/aaron/E/HRNet-Semantic-Segmentation-HRNet-OCR/output/building_patches')
    parser.add_argument('--maskdir', type=str, default='/media/aaron/E/HRNet-Semantic-Segmentation-HRNet-OCR/output/masks')
    parser.add_argument("--local_rank", type=int, default=-1)  
    args = parser.parse_args()
    
    dirpath = os.path.dirname(args.imgpairfile)
    
    
    istest = True 
    
  
    cudnn.benchmark = True 
    cudnn.deterministic = True 
    cudnn.enabled = True 
    

    device = torch.device('cuda:{}'.format(args.local_rank))    
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", init_method="env://") 
       
    model = DISTS()
    if istest:
        print("Loading optimal model parameters ......")
        pretrained_dict = torch.load(os.path.join(dirpath, 'best.pth'))
        model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.startswith('module')})
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=True,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )
    
    #Prepare data.
    sdataset = StreetscapeDataset(args.imgpairfile, args.imgdir, args.maskdir, istest=istest)
    ssampler = torch.utils.data.distributed.DistributedSampler(sdataset, shuffle=False) 
    
    
    sloader = DataLoader(
        sdataset,
        batch_size=10 if istest else 4,
        shuffle=False,
        num_workers=4 if istest else 0,
        pin_memory=True,
        sampler=ssampler if istest else None
        ) 
        
        
    
    if istest:       
        model.eval()
        rk = args.local_rank
        df = pd.DataFrame(columns=['ID', 'label'])
        with torch.no_grad():
            for (ind, x, y, mask, target) in tqdm(sloader):
                x = x.cuda()
                y = y.cuda()
                mask = mask.cuda()
                target = target.cuda()
                score = model(x, y, mask, target, istest=True) 
                label = torch.argmax(score.detach().cpu(), 1, keepdim=True) * 2 - 1 #Find the label with maximum score and change it to [-1, 1] from [0, 1]                              
                tempdf = torch.cat((torch.unsqueeze(ind.float(), 1), label.float()),1)   
                tempdf = pd.DataFrame(tempdf.numpy(), columns=['ID', 'label'])   
                df = pd.concat([df, tempdf])    
        
        df['ID'] = df['ID'].astype(int) 
        df['label'] = df['label'].astype(int) 
        df.to_csv("/media/aaron/E/DPSM"+str(rk)+".csv", index = False)
        print("Testing/inference task of GPU " + str(rk) + " is done!")   
