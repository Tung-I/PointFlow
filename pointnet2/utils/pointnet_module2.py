import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from pointnet2.utils import pointnet2_utils as pointutils


class RSSAModule(nn.Module): # set abstraction
    def __init__(self, npoint, radius, nsample, c_in, c_out, first_layer):
        super(RSSAModule, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        # self.group_all = group_all
        # self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        # last_channel = in_channel+3   # TODO：
        # for out_channel in mlp:
        #     self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
        #     self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        #     last_channel = out_channel
        

        self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)
        self.first_layer = first_layer

        # RS
        c_in = c_in + 3
        if first_layer:
            self.mapping_func1 = nn.Conv2d(in_channels=10, out_channels=math.floor(c_out / 2), kernel_size=(1, 1), stride=(1, 1), bias=True)
            self.bn_mapping = nn.BatchNorm2d(math.floor(c_out / 2))
            self.mapping_func2 = nn.Conv2d(in_channels=math.floor(c_out / 2), out_channels=16, kernel_size=(1, 1), stride=(1, 1), bias=True)
            self.xyz_raising = nn.Conv2d(in_channels=c_in, out_channels=16, kernel_size=(1, 1), stride=(1, 1), bias=True)
            self.bn_xyz_raising = nn.BatchNorm2d(16)
            self.bn_rsconv = nn.BatchNorm2d(16)
            self.cr_mapping = nn.Conv1d(in_channels=16, out_channels=c_out, kernel_size=1, stride=1, bias=True)
            self.bn_channel_raising = nn.BatchNorm1d(c_out)
        else:
            self.mapping_func1 = nn.Conv2d(in_channels=10, out_channels=math.floor(c_out / 4), kernel_size=(1, 1), stride=(1, 1), bias=True)
            self.bn_mapping = nn.BatchNorm2d(math.floor(c_out / 4))
            self.mapping_func2 = nn.Conv2d(in_channels=math.floor(c_out / 4), out_channels=c_in, kernel_size=(1, 1), stride=(1, 1), bias=True)
            self.bn_rsconv = nn.BatchNorm2d(c_in)
            self.cr_mapping = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1, stride=1, bias=True)
            self.bn_channel_raising = nn.BatchNorm1d(c_out)


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()   # [B, N, C]

        fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, npoint]
        new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, 3, npoint]
        new_xyz_t = new_xyz.permute(0, 2, 1).contiguous()

        _, idx = pointutils.knn(self.nsample, new_xyz_t, xyz_t)   # [B, npoint, nsample]
        neighbors = pointutils.grouping_operation(xyz, idx)   # [B, 3, npoint, nsample]
        centers = new_xyz.view(B, -1, self.npoint, 1).repeat(1, 1, 1, self.nsample)   # [B, 3, npoint, nsample]
        pos_diff = centers - neighbors  # [B, 3, npoint, nsample]
        distances = torch.norm(pos_diff, p=2, dim=1, keepdim=True)   # [B, 1, npoint, nsample]
        h_xi_xj = torch.cat([distances, pos_diff, centers, neighbors], dim=1)   # [B, 1+3+3+3, npoint, nsample]

        x = pointutils.grouping_operation(points, idx)   # [B, D, npoint, nsample]
        x = torch.cat([neighbors, x], dim=1)   # [B, D+3, npoint, nsample]

        h_xi_xj = self.mapping_func2(F.relu(self.bn_mapping(self.mapping_func1(h_xi_xj))))   # [B, c_in, npoint, nsample]
        if self.first_layer:
            x = F.relu(self.bn_xyz_raising(self.xyz_raising(x)))   # [B, c_in, npoint, nsample]
        x = F.relu(self.bn_rsconv(torch.mul(h_xi_xj, x)))   # (B, c_in, npoint, nsample)
        x = torch.max(x, -1)[0]   # [B, c_in, npoint]
        x = F.relu(self.bn_channel_raising(self.cr_mapping(x)))   # [B, c_out, npoint]


        return new_xyz, x


class SAModule(nn.Module): # set abstraction
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(SAModule, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel+3   # TODO：
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if group_all:
            self.queryandgroup = pointutils.GroupAll()
        else:
            self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()   # [B, N, C]

        if self.group_all == False:
            fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, npoint]
            new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, npoint]
        else:
            new_xyz = xyz
        new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points) # [B, D+C, npoint, nsample]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))   # [B, channel, npoint, nsample]
        
        new_points = torch.max(new_points, -1)[0]   # [B, channel, nsample]

        return new_xyz, new_points


class FEModule(nn.Module): # flow embedding
    def __init__(self, radius, nsample, in_channel, mlp, pooling='max', corr_func='concat', knn=True):
        super(FEModule, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if corr_func is 'concat':
            last_channel = in_channel*2+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            xyz1: (batch_size, 3, npoint)
            xyz2: (batch_size, 3, npoint)
            feat1: (batch_size, channel, npoint)
            feat2: (batch_size, channel, npoint)
        Output:
            xyz1: (batch_size, 3, npoint)
            feat1_new: (batch_size, mlp[-1], npoint)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, N, C = pos1_t.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)   # [B, N, S]
        else:
            idx = pointutils.ball_query(self.radius, self.nsample, pos2_t, pos1_t)
   
        pos2_grouped = pointutils.grouping_operation(pos2, idx) # [B, 3, N, S]
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B, 3, N, S]
        
        feat2_grouped = pointutils.grouping_operation(feature2, idx)  # [B, C, N, S]
        if self.corr_func=='concat':
            feat_diff = torch.cat([feat2_grouped, feature1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)], dim = 1)  # [B, 2*C, N, S]
        
        feat1_new = torch.cat([pos_diff, feat_diff], dim=1)  # [B, 2*C+3, N, S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat1_new = F.relu(bn(conv(feat1_new)))

        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]
        return pos1, feat1_new


class SUModule(nn.Module): # set upconv
    def __init__(self, nsample, radius, f1_channel, f2_channel, mlp, mlp2, knn=True):
        super(SUModule, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()

        last_channel = f2_channel+3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

        if len(mlp) is not 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel = last_channel + f1_channel

        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)
        Inputs:
            xyz1: (batch_size, 3, npoint1)
            xyz2: (batch_size, 3, npoint2)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers, more points)
            feat2: (batch_size, channel1, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)
            TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B,C,N = pos1.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)   # [B, N1, S]
        else:
            idx = pointutils.ball_query(self.radius, self.nsample, pos2_t, pos1_t)
        
        pos2_grouped = pointutils.grouping_operation(pos2, idx)
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)    # [B, 3, N1, S]

        feat2_grouped = pointutils.grouping_operation(feature2, idx)
        feat_new = torch.cat([feat2_grouped, pos_diff], dim = 1)   # [B, C1+3, N1, S]
        
        for conv in self.mlp1_convs:
            feat_new = conv(feat_new)

        # max pooling
        feat_new = feat_new.max(-1)[0]   # [B, mlp1[-1], N1]

        # concatenate feature in early layer
        if feature1 is not None:
            feat_new = torch.cat([feat_new, feature1], dim=1)   # [B, mlp1[-1]+feat1_channel, N1]

        for conv in self.mlp2_convs:
            feat_new = conv(feat_new)
        
        return feat_new


class FPModule(nn.Module): # feature propagation
    def __init__(self, in_channel, mlp):
        super(FPModule, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape
        
        # dists = square_distance(pos1, pos2)
        # dists, idx = dists.sort(dim=-1)
        # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
        dists, idx = pointutils.three_nn(pos1_t, pos2_t)   # [B, N, K=3]
        dists[dists < 1e-10] = 1e-10
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, -1, keepdim=True)
        interpolated_feat = torch.sum(pointutils.grouping_operation(feature2, idx) * weight.view(B, 1, N, 3), dim = -1) # [B, C, N, S=3] -> [B, C, N]
        
        if feature1 is not None:
            feat_new = torch.cat([interpolated_feat, feature1], 1)
        else:
            feat_new = interpolated_feat
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat_new = F.relu(bn(conv(feat_new)))
        return feat_new