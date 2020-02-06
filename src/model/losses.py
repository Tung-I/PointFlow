import torch
import torch.nn as nn
from pointnet2.utils import pointnet2_utils as pointutils


class MyL2Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        flow = target['flow']
        flow_pred = output['flow']

        # minimum = torch.min(flow)
        # maximum = torch.max(flow)
        # output = (output - minimum) / (maximum - minimum)
        # flow = (flow - minimum) / (maximum - minimum)
        if mask is not None:
            err = torch.norm(flow - flow_pred, p=2, dim=1)
            err = err * mask
            loss = torch.sum(err) / (torch.sum(mask) + 1e-20)
        else:
            loss = torch.norm(flow - flow_pred, p=2, dim=1).mean()

        return loss

class SamplingLoss1(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        c1 = output['l1_c1'] # [B, 3, N]
        dist = torch.norm(c1, p=2, dim=1).mean()
        threshold = 1.0
        if dist <= threshold:
            loss = threshold - dist
        else:
            loss = dist * 0.
        
        return loss
        # return 1.0 - c1_dist

class SamplingLoss2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        c1 = output['l2_c1'] # [B, 3, N]
        dist = torch.norm(c1, p=2, dim=1).mean()
        threshold = 1.0
        if dist <= threshold:
            loss = threshold - dist
        else:
            loss = dist * 0.
        
        return loss

# class ChamferLoss(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, output, target, mask=None):
#         pos1 = target['pc2']
#         # feat1 = output['feat']
#         pos2 = output['pc1_warped']
#         # feat2 = target['feat']

#         B, C, N = pos1.size()
#         pos1_t = pos1.permute(0, 2, 1).contiguous()
#         pos2_t = pos2.permute(0, 2, 1).contiguous()


#         pos_diff_forward, idx = pointutils.knn(1, pos1_t, pos2_t)  # [B, N, 1]
#         pos_diff_backward, idx = pointutils.knn(1, pos2_t, pos1_t)  # [B, N, 1]

#         loss = torch.sum(pos_diff_forward, 1).mean() + torch.sum(pos_diff_backward, 1).mean()

#         return loss


# class SmoothnessLoss(nn.Module):

#     def __init__(self, radius, nsample):
#         super().__init__()
#         self.radius = radius
#         self.nsample = nsample
#         # self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

#     def forward(self, output, target, mask=None):
#         pos1 = target['pc1']
#         flow = output['flow']

#         pos1_t = pos1.permute(0, 2, 1).contiguous()

#         idx = pointutils.ball_query(self.radius, self.nsample, pos1_t, pos1_t)  # (B, N, nsample)
#         grouped_flow = pointutils.grouping_operation(flow, idx)  # (B, 3, N, nsample)

#         B, C, N = flow.size()
#         flow_diff = grouped_flow - flow.view(B, -1, N, 1)  # (B, 3, N, nsample)

#         loss = torch.sum(torch.norm(flow_diff, p=2, dim=1), 1)  # (B, nsample)
#         loss = loss.mean()

#         # loss = torch.norm(flow_diff, p=2, dim=1).mean()
#         return loss






