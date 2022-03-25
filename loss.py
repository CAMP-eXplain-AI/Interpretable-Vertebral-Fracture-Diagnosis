from typing import Optional
import torch
import torch.nn.functional as F

def get_breadstick_probabilities(logits):
    # logits to linear projectsion to probabilities 
    projections = torch.sigmoid(logits)
    batch_size, num_projections = projections.shape
    num_classes = num_projections + 1
    probs = torch.zeros(batch_size, num_classes, dtype=projections.dtype, device=projections.device)
    prob = 0
    for i in range(projections.shape[-1]):
        prob = projections[:,i] * (1 - prob)
        probs[:,i] = prob
    probs[:, -1] = (1 - projections.select(-1, -1)) * prob
    return probs

def ordinal_regression_loss(logits, targets, tao=1, eta=0.15, class_weights=None):
    '''
    ordinal regression loss based on this paper:
    Liu, Xiaofeng, et al. "Unimodal regularized neuron stick-breaking for ordinal classification." Neurocomputing 388 (2020): 34-44.

    it is important to have N-1 logits with N classes

    '''
    probs = get_breadstick_probabilities(logits)
    batch_size, num_classes = probs.shape
    class_weights = class_weights if class_weights is not None else torch.ones(num_classes, device=logits.device)

    # cross entropy loss with unimodal regularization
    # distribution of normalized exponential function

    q = torch.softmax(
        torch.exp(
            -torch.abs(
                torch.arange(num_classes, device=targets.device).repeat(batch_size, 1) - targets.reshape(batch_size,1)
            ) / tao
        ), dim=-1
    )
    # smooth distribution with eta
    
    q = (1 - eta) * q + eta / num_classes
    loss = torch.sum(class_weights * q * -torch.log(probs), dim=-1).mean()

    return loss

# Adapted from https://github.com/kornia/kornia/blob/c2273bbfe152c86a48923e473a37c05e28f7fe43/kornia/losses/focal.py
def focal_loss(input: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: Optional[float] = 2.0) -> torch.Tensor:
    """Criterion that computes Focal loss.
    According to Lin, Tsung-Yi, et al. "Focal Loss for Dense Object Detection". Proceedings of the IEEE international conference on computer vision (2017): 2980-2988,
    the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    """
    assert input.size(0) == target.size(0)

    if len(input.shape) == 1:
        # binary focal loss
        probs_pos = torch.sigmoid(input)
        probs_neg = torch.sigmoid(-input)
        loss_tmp = -alpha * torch.pow(probs_neg, gamma) * target * F.logsigmoid(input) - (1 - alpha) * torch.pow(probs_pos, gamma) * (1.0 - target) * F.logsigmoid(-input)

    else:
        assert len(input.shape) >= 2

        assert target.size()[1:] == input.size()[2:]

        # compute softmax over the classes axis
        input_soft: torch.Tensor = F.softmax(input, dim=1)
        log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

        # create the labels one hot tensor
        one_hot = torch.zeros((target.shape[0], input.shape[1]) + target.shape[1:], device=input.device, dtype=input.dtype)
        target_one_hot: torch.Tensor = one_hot.scatter_(1, target.unsqueeze(1), 1.0) + 1e-6

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, gamma)

        focal = -alpha * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    return loss_tmp.mean()