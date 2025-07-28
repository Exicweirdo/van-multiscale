# MADE: Masked Autoencoder for Distribution Estimation

import torch
from numpy import log, log2
from torch import nn
from typing import List, Tuple
from utils import default_dtype_torch


class ResBlock(nn.Module):
    def __init__(self, block):
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias, exclusive):
        super(MaskedLinear, self).__init__(in_channels * n, out_channels * n, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.exclusive = exclusive

        self.register_buffer("mask", torch.ones([self.n] * 2))
        if self.exclusive:
            self.mask = 1 - torch.triu(self.mask)
        else:
            self.mask = torch.tril(self.mask)
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return super(MaskedLinear, self).extra_repr() + ", exclusive={exclusive}".format(
            **self.__dict__
        )


class SparseMaskedLinear(nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int,
        bias,
        exclusive,
        degeneracy: List[Tuple] = None,
    ):
        super(SparseMaskedLinear, self).__init__(in_channels * n, out_channels * n, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.exclusive = exclusive

        self.register_buffer("mask", torch.ones([self.n] * 2))
        if self.exclusive:
            self.mask = 1 - torch.triu(self.mask)
        else:
            self.mask = torch.tril(self.mask)
        if degeneracy is not None:
            for index, length in degeneracy:
                self.mask[index : index + length, index : index + length] = torch.diag(
                    torch.diagonal(self.mask[index : index + length, index : index + length])
                )
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return super(SparseMaskedLinear, self).extra_repr() + ", exclusive={exclusive}".format(
            **self.__dict__
        )


# TODO: reduce unused weights, maybe when torch.sparse is stable


class ChannelLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias):
        super(ChannelLinear, self).__init__(in_channels * n, out_channels * n, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.register_buffer("mask", torch.eye(self.n))
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


class MADESimple1d(nn.Module):
    def __init__(self, **kwargs):
        super(MADESimple1d, self).__init__()
        self.n = kwargs["n"]
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
            self.register_buffer("x_hat_mask", torch.ones([self.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer("x_hat_bias", torch.zeros([self.L] * 2))
            self.x_hat_bias[0, 0] = 0.5

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
        self.net = nn.Sequential(*layers)

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
        x = x.view(x.shape[0], -1)
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

        return x_hat

    # sample = +/-1, +1 = up = white, -1 = down = black
    # sample.dtype == default_dtype_torch
    # x_hat = p(x_{i} == +1 | x_{0}, ..., x_{i - 1})
    # 0 < x_hat < 1
    # x_hat will not be flipped by z2
    def sample(self, batch_size):
        sample = torch.zeros([batch_size, self.n], dtype=default_dtype_torch, device=self.device)
        # This procedure seems slow (O(n^2))
        for i in range(self.n):
            x_hat = self.forward(sample)
            sample[:, i] = torch.bernoulli(x_hat[:, i]).to(default_dtype_torch) * 2 - 1

        if self.z2:
            # Binary random int 0/1
            flip = (
                torch.randint(2, [batch_size, 1], dtype=sample.dtype, device=sample.device) * 2 - 1
            )
            sample *= flip

        return sample, x_hat

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


class MADEnew1d(nn.Module):
    def __init__(self, **kwargs):
        super(MADEnew1d, self).__init__()
        self.n = kwargs["n"]
        self.net_depth = kwargs["net_depth"]
        self.net_width = kwargs["net_width"]
        self.bias = kwargs["bias"]
        self.z2 = kwargs["z2"]
        self.res_block = kwargs["res_block"]
        self.x_hat_clip = kwargs["x_hat_clip"]
        self.epsilon = kwargs["epsilon"]
        self.device = kwargs["device"]

        self.degeneracy = [(2**i, 2**i) for i in range(0, int(log2(self.n)))]
        assert log2(self.n) % 1 == 0, "n must be power of 2"
        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer("x_hat_mask", torch.ones([self.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer("x_hat_bias", torch.zeros([self.L] * 2))
            self.x_hat_bias[0, 0] = 0.5

        layers = []

        layers.append(
            SparseMaskedLinear(
                1,
                1 if self.net_depth == 1 else self.net_width,
                self.n,
                self.bias,
                exclusive=True,
                degeneracy=self.degeneracy,
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
        self.net = nn.Sequential(*layers)

    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(
            SparseMaskedLinear(
                in_channels,
                out_channels,
                self.n,
                self.bias,
                exclusive=False,
                degeneracy=self.degeneracy,
            )
        )
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(ChannelLinear(in_channels, out_channels, self.n, self.bias))
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(
            MaskedLinear(
                in_channels,
                out_channels,
                self.n,
                self.bias,
                exclusive=False,
                degeneracy=self.degeneracy,
            )
        )
        block = ResBlock(nn.Sequential(*layers))
        return block

    @staticmethod
    def fold(x: torch.Tensor, n: int) -> torch.Tensor:
        # x: (batch, n)
        x_list = [x[:, 0].view(x.size(0), -1)]
        for i in range(int(log2(n)) - 1, -1, -1):
            x_list.append(x[:, 2**i :: 2 ** (i + 1)].view(x.size(0), -1))
        return torch.cat(x_list, dim=1)

    @staticmethod
    def unfold(x: torch.Tensor, n: int) -> torch.Tensor:
        x_out = torch.zeros_like(x)
        x_out[:, 0] += x[:, 0]
        index = 1
        length = 1
        for i in range(int(log2(n)) - 1, -1, -1):
            x_out[:, 2**i :: 2 ** (i + 1)] += x[:, index : index + length]
            index += length
            length *= 2
        return x_out

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        x_fold = MADEnew1d.fold(x, self.n)
        x_hat_fold = self.net(x_fold)
        x_hat = MADEnew1d.unfold(x_hat_fold, self.n)
        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip, 1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        return x_hat

    def forward_onfolded(self, x_fold: torch.Tensor):
        x_fold = x_fold.view(x_fold.shape[0], -1)
        x_hat = self.net(x_fold)
        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip, 1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        return x_hat

    # sample = +/-1, +1 = up = white, -1 = down = black
    # sample.dtype == default_dtype_torch
    # x_hat = p(x_{i} == +1 | x_{0}, ..., x_{i - 1})
    # 0 < x_hat < 1
    # x_hat will not be flipped by z2
    def sample(self, batch_size):
        # sample is done on the folded space and then unfolded
        sample_fold = torch.zeros(
            [batch_size, self.n], dtype=default_dtype_torch, device=self.device
        )
        # This procedure seems slow (O(n^2))
        x_hat_0 = self.forward_onfolded(sample_fold)
        sample_fold[:, 0] = torch.bernoulli(x_hat_0[:, 0]).to(default_dtype_torch) * 2 - 1
        for index, length in self.degeneracy:
            x_hat = self.forward_onfolded(sample_fold)
            sample_fold[:, index : index + length] = (
                torch.bernoulli(x_hat[:, index : index + length]).to(default_dtype_torch) * 2 - 1
            )

        sample = MADEnew1d.unfold(sample_fold, self.n)
        if self.z2:
            # Binary random int 0/1
            flip = (
                torch.randint(2, [batch_size, 1], dtype=sample.dtype, device=sample.device) * 2 - 1
            )
            sample *= flip

        return sample, MADEnew1d.unfold(x_hat, self.n)

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
