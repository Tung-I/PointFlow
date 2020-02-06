import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from src.model.nets.base_net import BaseNet
from pointnet2.utils.pointnet_module2 import CLModule, SAModule, FEModule, SUCModule, FPModule



class CLFlowNet3D(BaseNet):
    def __init__(self):
        super(CLFlowNet3D,self).__init__()

        self.cl1 = CLModule(npoint=1024, radius=0.5, nsample=32, in_channel=3, mlp1=[32, 32], mlp2=[16])
        self.cl2 = CLModule(npoint=256, radius=1.0, nsample=32, in_channel=64, mlp1=[64, 32], mlp2=[16])
        # self.cl3 = CLModule(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp1=[128, 64], mlp2=[16])
        # self.cl4 = CLModule(npoint=16, radius=4.0, nsample=8, in_channel=256, mlp1=[256, 64], mlp2=[16])

        self.sa1 = SAModule(npoint=1024, radius=0.5, nsample=16, in_channel=3, mlp=[32, 32, 64])
        self.sa2 = SAModule(npoint=256, radius=1.0, nsample=16, in_channel=64, mlp=[64, 64, 128])
        self.sa3 = SAModule(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256])
        self.sa4 = SAModule(npoint=16, radius=4.0, nsample=8, in_channel=256, mlp=[256 ,256, 512])
        
        self.fe_layer = FEModule(radius=10.0, nsample=64, in_channel = 128, mlp=[128, 128, 128], pooling='max', corr_func='concat')
        
        self.su1 = SUCModule(nsample=8, radius=2.4, f1_channel = 256, f2_channel = 512, mlp=[], mlp2=[256, 256])
        self.su2 = SUCModule(nsample=8, radius=1.2, f1_channel = 128+128, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.su3 = SUCModule(nsample=8, radius=0.6, f1_channel = 64, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.fp = FPModule(in_channel = 256+3, mlp = [256, 256])
        
        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2=nn.Conv1d(128, 3, kernel_size=1, bias=True)
        
    def forward(self, pc1, pc2, feature1, feature2):
        l1_c1, pc1_cen = self.cl1(pc1, feature1)
        l1_pc1, l1_feature1 = self.sa1(pc1, feature1, pc1_cen)
        l2_c1, l1_pc1_cen = self.cl2(l1_pc1, l1_feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1, l1_pc1_cen)

        l1_c2, pc2_cen = self.cl1(pc2, feature2)
        l1_pc2, l1_feature2 = self.sa1(pc2, feature2, pc2_cen)
        l2_c2, l2_pc2_cen = self.cl2(l1_pc2, l1_feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2, l2_pc2_cen)
        
        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        # l2_pc1_cen = self.cl3(l2_pc1, l2_feature1_new)
        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        # l3_pc1_cen = self.cl4(l3_pc1, l3_feature1)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)
        
        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        
        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        flow = self.conv2(x)

        return {'flow': flow, 'l1_c1':l1_c1, 'l2_c1':l2_c1}


        # l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        # l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)
        
        # l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        # l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        
        # _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        # l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        # l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)
        
        # l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        # l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        # l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        # l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        
        # x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        # flow = self.conv2(x)

        # return {'flow': flow}