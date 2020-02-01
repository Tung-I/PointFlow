import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import label


def epe_acc_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 1) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 1) + 1e-20) # B,N

    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2


class EPE(nn.Module):
    """The End Point Error.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.

        flow = target['flow']
        flow_pred = output['flow']

        # if mask is not None:
        #     epe = torch.norm(flow - flow_pred, p=2, dim=1)
        #     epe = epe * mask
        #     epe = torch.sum(epe) / (torch.sum(mask) + 1e-20)
        # else:
        #     epe = torch.norm(flow - flow_pred, p=2, dim=1).mean()

        flow = flow.detach().cpu().numpy()
        flow_pred = flow_pred.detach().cpu().numpy()
        mask = mask.cpu().numpy()

        epe, _, _ = epe_acc_np(flow_pred, flow, mask)

        return epe


class ACC_005(nn.Module):
    """The accuracy
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        flow = target['flow']
        flow_pred = output['flow']
        
        flow = flow.detach().cpu().numpy()
        flow_pred = flow_pred.detach().cpu().numpy()
        mask = mask.cpu().numpy()

        _, acc1, _ = epe_acc_np(flow_pred, flow, mask)

        return acc1


class ACC_01(nn.Module):
    """The accuracy
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        flow = target['flow']
        flow_pred = output['flow']
        
        flow = flow.detach().cpu().numpy()
        flow_pred = flow_pred.detach().cpu().numpy()
        mask = mask.cpu().numpy()

        _, _, acc2 = epe_acc_np(flow_pred, flow, mask)

        return acc2


class ADE(nn.Module):
    """The accuracy
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        flow = target['flow']
        flow_pred = output['flow']
        # flow_len = torch.norm(flow, p=2, dim=1)
        # flow_pred_len = torch.norm(flow_pred, p=2, dim=1)
        # ade = (flow * flow_pred).sum(1) / (flow_len * flow_pred_len) # (B, N)
        cos = nn.CosineSimilarity(dim=1, eps=1e-10)
        ade = cos(flow, flow_pred) # (B, N)

        if mask is not None:
            ade = (ade * mask)
            ade = ade.sum(1) / (mask.sum(1) + 1e-10)
            ade = ade.mean()
        else:
            ade = ade.mean()

        return torch.acos(ade) * 180 / 3.14159


class F1Score(nn.Module):
    """The accuracy
    """
    def __init__(self, threshold):
        super().__init__()
        self.th = threshold

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        flow = target['flow']
        flow_pred = output['flow']
        err = torch.norm(flow_pred-flow, p=2, dim=1)
        flow_len = torch.norm(flow, p=2, dim=1)

        if mask is not None:
            f1 = ((err/(flow_len+1e-20) <= self.th) * mask.byte())
            f1 = torch.sum(f1.float())
            f1 = f1 / (torch.sum(mask) + 1e-20)
        else:
            f1 = (err/(flow_len+1e-20) <= self.th).byte()
            f1 = torch.mean(f1.float())

        return f1




