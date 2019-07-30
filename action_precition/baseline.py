# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:09:49 2019

@author: epyir
"""
import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def forward(self, input, target):
        pass

class baseline(nn.Module):
    """
    This baseline simply add/cat outputs of ROI 
    roi_num = # proposals
    class_num = # actions
    """
    def __init__(self, roi_num=1000, class_num=4, is_cat=False, is_feature=False):
        super(baseline, self).__init__()
        self.roi_num = roi_num # number of proposals
        self.class_num = class_num # number of actions
        self.is_cat = is_cat
        self.is_feature = is_feature
        
        if is_cat:
            # cat
            self.fc1 = nn.Linear(roi_num, 256)
        else:
            # else add
            self.fc1 = nn.Linear(256*7*7, 256)
        self.relu1 = nn.PReLU(256)
        # self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, self.class_num)

        self.Pool = nn.AvgPool2d((184,320))
        self.sftx1 = nn.Softmax(dim=1)
        self.sttx2 = nn.Softmax(dim=1)

    def forward(self, x, y):
        """
        x (Tensor) is the feature from ROI pooling/align. e.g. x.size = (1000, 256, 7, 7)
        """
        if not self.is_cat:
            x = x.sum(dim=0).unsqueeze(0) # addition over channels, x.shape = (1,256,7,7)
        else:
            x = x.reshape(-1) # under construction
        x = x.view(x.size(0), -1) # Size (1,256*7*7)
        y = self.Pool(y) # Size(1, 256, 1, 1)
        y = y.view(y.size(0), -1) # Size(1, 256)
        x = self.fc1(x) # Size (1,256)
        # x = self.bn1(x)
        # x = self.bn1(self.fc1(x))
        x = self.relu1(x) # Size (1, 256)

        # normalize features
        x = self.sftx1(x)
        y = self.sttx2(y)


        # concat 2 features
        x = torch.cat((x, y), 1) # Size(1,512)

        x = self.fc2(x) # Size (1, 4)
        return x
    
def main():
    """
    Just a test.
    """
    x = torch.ones(1000,256,7,7)
    net = baseline()
    out = net(x)
    print(out)
    print(out.shape)

if __name__ == "__main__":
    main()
