import torch
import torch.nn as nn
import torch.nn.functional as F


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    Multilabel categorical crossentroy
    Description: y_ True and Y_ The shape of PRED is consistent, y_ The element of true is either 0 or 1,1 indicates that the corresponding class is the target class, and 0 indicates that the corresponding class is a non target class.
    Warning: please ensure Y_ The value range of PRED is all real numbers, in other words, in general, y_ pred.Do not add activation functions, especially sigmoid or softmax! forecast.Phase outputs y_ Classes with PRED greater than 0. If you have any questions, please read and understand them carefully this paper.
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros],dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros),dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):

        loss = multilabel_categorical_crossentropy(labels,logits)
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = torch.zeros_like(logits[..., :1])
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output[:,1:].sum(1) == 0.).to(logits)

        return output

