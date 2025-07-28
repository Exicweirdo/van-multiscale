import torch
from torch import nn
from numpy import log

from .special_layer import MaskedLinear, ResBlock, ChannelLinear
from utils import default_dtype_torch


class Made(nn.Module):
    def __init__(self, **kwargs):
        super(Made, self).__init__()
        self.L = kwargs["L"]
        self.n = self.L**2
        self.net_depth = kwargs["net_depth"]
        self.net_width = kwargs["net_width"]
        self.bias = kwargs["bias"]
        self.z2 = kwargs["z2"]
        self.res_block = kwargs["res_block"]
        self.x_hat_clip = kwargs["x_hat_clip"]
        self.epsilon = kwargs["epsilon"]
        self.device = kwargs["device"]

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer("x_hat_mask", torch.ones([self.L**2]))
            self.x_hat_mask[0] = 0
            self.register_buffer("x_hat_bias", torch.zeros([self.L**2]))
            self.x_hat_bias[0] = 0.5

        layers = []

        layers.append(
            MaskedLinear(
                1,
                1 if self.net_depth == 1 else self.net_width,
                self.n,
                self.bias,
                exclusive=True,
            )
        )

        for count in range(self.net_depth - 2):

            if self.res_block:
                layers.append(self._build_res_block(self.net_width, self.net_width))
            else:
                layers.append(self._build_simple_block(self.net_width, self.net_width))

        if self.net_depth >= 2:
            layers.append(self._build_simple_block(self.net_width, 1))

        layers.append(nn.Sigmoid())
        self.net = torch.compile(nn.Sequential(*layers))

    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(MaskedLinear(in_channels, out_channels, self.n, self.bias, exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(ChannelLinear(in_channels, out_channels, self.n, self.bias))
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(MaskedLinear(in_channels, out_channels, self.n, self.bias, exclusive=False))
        block = ResBlock(nn.Sequential(*layers))
        return block

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], -1)
        x_hat = self.net(x)
        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip, 1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        return x_hat.view(x_hat.shape[0], self.L, self.L)

    # sample = +/-1, +1 = up = white, -1 = down = black
    # sample.dtype == default_dtype_torch
    # x_hat = p(x_{i} == +1 | x_{0}, ..., x_{i - 1})
    # 0 < x_hat < 1
    # x_hat will not be flipped by z2
    @torch.compile
    def sample(self, batch_size):
        sample = torch.zeros([batch_size, self.n], dtype=default_dtype_torch, device=self.device)
        # This procedure seems slow (O(n^2))
        for i in range(self.n):
            x_hat = self.forward(sample)
            x_hat = x_hat.view(x_hat.shape[0], -1)
            sample[:, i] = torch.bernoulli(x_hat[:, i]).to(default_dtype_torch) * 2 - 1

        if self.z2:
            # Binary random int 0/1
            flip = (
                torch.randint(2, [batch_size, 1], dtype=sample.dtype, device=sample.device) * 2 - 1
            )
            sample *= flip

        return sample.view(sample.shape[0], self.L, self.L), x_hat.view(
            x_hat.shape[0], self.L, self.L
        )

    def partial_sample(self, batch_size, sample_init, start):
        sample = sample_init.detach().clone().reshape(batch_size, -1)
        for i in range(start, self.n):
            x_hat = self.forward(sample)
            x_hat = x_hat.view(x_hat.shape[0], -1)
            sample[:, i] = torch.bernoulli(x_hat[:, i]).to(default_dtype_torch) * 2 - 1

        if self.z2:
            # Binary random int 0/1
            flip = (
                torch.randint(2, [batch_size, 1], dtype=sample.dtype, device=sample.device) * 2 - 1
            )
            sample *= flip

        return sample.view(sample.shape[0], self.L, self.L), x_hat.view(
            x_hat.shape[0], self.L, self.L
        )
        
    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (
            1 - mask
        )
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.z2:
            # Density estimation on inverted sample
            sample_inv = -sample
            x_hat_inv = self.forward(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob
