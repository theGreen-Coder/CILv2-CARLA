import torch
import torch.nn as nn
from configs import g_conf
import network.adaptative_robust_loss
import numpy as np
import wandb

from network.adaptative_robust_loss import distribution
from network.adaptative_robust_loss import util
from network.adaptative_robust_loss import wavelet

class Action_nospeed_L1(nn.Module):
    def __init__(self):
        super(Action_nospeed_L1, self).__init__()

    def forward(self, params):
        B = params['action_output'].shape[0]  # batch_size

        diff = params['action_output'][:,-1,:] - params['targets_action'][-1]

        # SingleFrame model - we only take into account the last frame's action
        actions_loss_mat = torch.abs(diff)  # (B, 2)

        steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
        steer_loss = torch.sum(steer_loss) / B

        if g_conf.ACCELERATION_AS_ACTION:
            acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
            acceleration_loss = torch.sum(acceleration_loss) / B

            loss = steer_loss + acceleration_loss

            return loss, steer_loss, acceleration_loss, diff

        else:
            throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
            brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
            throttle_loss = torch.sum(throttle_loss) / B
            brake_loss = torch.sum(brake_loss) / B

            loss = steer_loss + throttle_loss + brake_loss

            return loss, steer_loss, throttle_loss, brake_loss, diff

class Adaptative_Robust_Loss(nn.Module):
    def __init__(self, num_dims, float_dtype, device, alpha_lo=0.001, alpha_hi=1.999, alpha_init=None, scale_lo=1e-5, scale_init=1.0):
        super(Adaptative_Robust_Loss, self).__init__()
        self.diff = []

        if not np.isscalar(alpha_lo):
            raise ValueError('`alpha_lo` must be a scalar, but is of type {}'.format(type(alpha_lo)))
        if not np.isscalar(alpha_hi):
            raise ValueError('`alpha_hi` must be a scalar, but is of type {}'.format(type(alpha_hi)))
        if alpha_init is not None and not np.isscalar(alpha_init):
            raise ValueError('`alpha_init` must be None or a scalar, but is of type {}'.format(type(alpha_init)))
        if not alpha_lo >= 0:
            raise ValueError('`alpha_lo` must be >= 0, but is {}'.format(alpha_lo))
        if not alpha_hi >= alpha_lo:
            raise ValueError('`alpha_hi` = {} must be >= `alpha_lo` = {}'.format(
            alpha_hi, alpha_lo))
        if alpha_init is not None and alpha_lo != alpha_hi:
            if not (alpha_init > alpha_lo and alpha_init < alpha_hi):
                raise ValueError('`alpha_init` = {} must be in (`alpha_lo`, `alpha_hi`) = ({} {})'.format(alpha_init, alpha_lo, alpha_hi))
        if not np.isscalar(scale_lo):
            raise ValueError('`scale_lo` must be a scalar, but is of type {}'.format(type(scale_lo)))
        if not np.isscalar(scale_init):
            raise ValueError('`scale_init` must be a scalar, but is of type {}'.format(type(scale_init)))
        if not scale_lo > 0:
            raise ValueError('`scale_lo` must be > 0, but is {}'.format(scale_lo))
        if not scale_init >= scale_lo:
            raise ValueError('`scale_init` = {} must be >= `scale_lo` = {}'.format(scale_init, scale_lo))

        self.num_dims = num_dims
        if float_dtype == np.float32:
            float_dtype = torch.float32
        if float_dtype == np.float64:
            float_dtype = torch.float64
        self.float_dtype = float_dtype
        self.device = device
        if isinstance(device, int) or (isinstance(device, str) and 'cuda' in device) or (isinstance(device, torch.device) and device.type == 'cuda'):
            torch.cuda.set_device(self.device)

        self.distribution = distribution.Distribution()

        if alpha_lo == alpha_hi:
            # If the range of alphas is a single item, then we just fix `alpha` to be
            # a constant.
            self.fixed_alpha = torch.tensor(
            alpha_lo, dtype=self.float_dtype,
            device=self.device)[np.newaxis, np.newaxis].repeat(1, self.num_dims)
            self.alpha = lambda: self.fixed_alpha
        else:
            # Otherwise we construct a "latent" alpha variable and define `alpha`
            # As an affine function of a sigmoid on that latent variable, initialized
            # such that `alpha` starts off as `alpha_init`.
            if alpha_init is None:
                alpha_init = (alpha_lo + alpha_hi) / 2.
            latent_alpha_init = util.inv_affine_sigmoid(
                alpha_init, lo=alpha_lo, hi=alpha_hi)
            self.register_parameter(
                'latent_alpha',
                torch.nn.Parameter(
                    latent_alpha_init.clone().detach().to(
                        dtype=self.float_dtype,
                        device=self.device)[np.newaxis, np.newaxis].repeat(
                            1, self.num_dims),
                    requires_grad=True))
            self.alpha = lambda: util.affine_sigmoid(
                self.latent_alpha, lo=alpha_lo, hi=alpha_hi)

        if scale_lo == scale_init:
            # If the difference between the minimum and initial scale is zero, then
            # we just fix `scale` to be a constant.
            self.fixed_scale = torch.tensor(
                scale_init, dtype=self.float_dtype,
                device=self.device)[np.newaxis, np.newaxis].repeat(1, self.num_dims)
            self.scale = lambda: self.fixed_scale
        else:
            # Otherwise we construct a "latent" scale variable and define `scale`
            # As an affine function of a softplus on that latent variable.
            self.register_parameter(
                'latent_scale',
                torch.nn.Parameter(
                    torch.zeros((1, self.num_dims)).to(
                        dtype=self.float_dtype, device=self.device),
                    requires_grad=True))
            self.scale = lambda: util.affine_softplus(
                self.latent_scale, lo=scale_lo, ref=scale_init)

    def lossfun(self, x, **kwargs):
        """Computes the loss on a matrix.

        Args:
        x: The residual for which the loss is being computed. Must be a rank-2
            tensor, where the innermost dimension is the batch index, and the
            outermost dimension must be equal to self.num_dims. Must be a tensor or
            numpy array of type self.float_dtype.
        **kwargs: Arguments to be passed to the underlying distribution.nllfun().

        Returns:
        A tensor of the same type and shape as input `x`, containing the loss at
        each element of `x`. These "losses" are actually negative log-likelihoods
        (as produced by distribution.nllfun()) and so they are not actually
        bounded from below by zero. You'll probably want to minimize their sum or
        mean.
        """
        x = torch.as_tensor(x)
        assert len(x.shape) == 2
        assert x.shape[1] == self.num_dims
        assert x.dtype == self.float_dtype
        return self.distribution.nllfun(x, self.alpha(), self.scale(), **kwargs)

    def forward(self, params):
        B = params['action_output'].shape[0]  # batch_size

        diff = params['action_output'][:,-1,:] - params['targets_action'][-1]

        # SingleFrame model - we only take into account the last frame's action
        actions_loss_mat = self.lossfun(diff)  # (B, 2)

        steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
        steer_loss = torch.sum(steer_loss) / B

        if g_conf.ACCELERATION_AS_ACTION:
            acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
            acceleration_loss = torch.sum(acceleration_loss) / B

            loss = steer_loss + acceleration_loss

            return loss, steer_loss, acceleration_loss, diff

        else:
            throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
            brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
            throttle_loss = torch.sum(throttle_loss) / B
            brake_loss = torch.sum(brake_loss) / B

            loss = steer_loss + throttle_loss + brake_loss

            return loss, steer_loss, throttle_loss, brake_loss, diff

def Loss(loss, device):

    if loss=='Action_nospeed_L1':
        return Action_nospeed_L1()

    elif loss=='Adaptative_Robust_Loss':
        return Adaptative_Robust_Loss(num_dims = 2, alpha_lo=0, alpha_hi=2, alpha_init=1, float_dtype=np.float32, device=device)

    else:
        raise NotImplementError(" The loss of this model type has not yet defined ")
