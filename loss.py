import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import math
import numpy as np

class ScoreTripletLoss(Module):
    def __init__(self, margin=1.0, s=64.0, rank_threshold=50):
        super(ScoreTripletLoss, self).__init__()
        self.margin = margin
        self.s = s  # scalar value default is 64, similar to adaface
        self.rank_threshold = rank_threshold
        assert self.rank_threshold > 1, 'rank threshold should be greater than 1'
        print(f'using score triplet loss with margin: {margin}, s: {s}, rank_threshold: {rank_threshold}')

    def cal_rank_loss_cc(self, pred_weights, scores, labels, center_labels):
        # scores: (B, N), labels: (B,), center_labels: (N,)
        mask = (labels.unsqueeze(1) == center_labels.unsqueeze(0))
        ranks = torch.zeros_like(labels)
        for i in range(labels.size(0)):
            # Get the scores for the current label
            current_scores = scores[i]
            # Get the mask for the current label
            current_mask = mask[i]
            # Get the sorted indices for the current scores
            sorted_indices = current_scores.argsort(descending=True)
            # Find the rank of the current sample in the center, use the maxium rank since there may be multiple same labels in the center
            rank = (sorted_indices.unsqueeze(1) == torch.nonzero(current_mask, as_tuple=True)[0]).nonzero(as_tuple=True)[0]
            if rank.numel() == 0:
                # If the current sample is not registed the center (happen in MEVID), set the rank to the maximum rank
                rank = len(center_labels) - 1
            else:
                rank = rank.min()
            ranks[i] = rank + 1  # ranks are 1-based
        adapted_weights = torch.nn.functional.relu((self.rank_threshold - ranks) / (self.rank_threshold - 1))
        rank_loss = nn.MSELoss()(pred_weights, adapted_weights.unsqueeze(-1))
        print(f'rank_res is: {ranks.cpu().numpy()}')
        print(f"pred weight is {pred_weights.detach().cpu().numpy().squeeze()}")
        print(f'psuedo weight is {adapted_weights.detach().cpu().numpy().squeeze()}')
        return rank_loss

    def cal_score_loss_cc(self, fuse_scores, labels, center_labels):
        """
        gallery that may contain same subject templates
        """
        B, N = fuse_scores.shape
        assert (B == len(labels) and N == len(center_labels)), 'score shape not match with labels'
        
        mask = (labels.unsqueeze(1) == center_labels.unsqueeze(0))
        
        match_score = fuse_scores[mask]
        non_match_score = fuse_scores[~mask]

        nonmat_loss = torch.relu(non_match_score).mean() 
        match_loss = torch.relu(self.margin - match_score).mean()
        
        score_loss = nonmat_loss + match_loss
        return score_loss

    def forward(self, res, labels, center_labels):
        # scores dim: (B, num_galleries), label dim: (B,), center_labels dim: (num_galleries,)
        # rank loss
        loss = {}

        loss['score_loss'] = self.cal_score_loss_cc(res['fuse_scores'], labels, center_labels)
        
        loss['total_loss'] = loss['score_loss']
        
        return loss


def cal_match_nonmat(score_mat, q_pids, g_pids):
    mask = q_pids.unsqueeze(1) == g_pids.unsqueeze(0)
    match_scores = score_mat[mask].flatten()
    nonmat_scores = score_mat[~mask].flatten()
    return nonmat_scores, match_scores

def find_matches(scores, labels_g, labels_p):
    """
        scores: [p, n_g, n_p]
        labels_g: [n_g]
        labels_p: [n_p]
    """
    # mask indicating matching gallery-probe pairs
    match_mask = labels_g.unsqueeze(1) == labels_p.unsqueeze(0)

    # use the match mask to extract the scores
    return scores[:, None, match_mask]  # Shape: [p, 1, n_matches]

def compute_distance(x, y):
    """
        x: [p, n_x, c]
        y: [p, n_y, c]
    """
    x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
    y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
    inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
    dist = x2 + y2 - 2 * inner
    dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
    return dist
