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


class MADESimple2d(nn.Module):
    def __init__(self, **kwargs):
        super(MADESimple2d, self).__init__()
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

        return x_hat.view(x_hat.shape[0], self.L, self.L)

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


class MADEnew2d(nn.Module):
    def __init__(self, **kwargs):
        super(MADEnew2d, self).__init__()
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

        self.degeneracy = [(2**i, 2**i) for i in range(0, int(log2(self.n)))]
        assert log2(self.L) % 1 == 0, "L must be power of 2"
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
        self.net = torch.compile(nn.Sequential(*layers))

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
            SparseMaskedLinear(
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
    def fold(x: torch.Tensor, L: int) -> torch.Tensor:
        # x: (batch, L, L)
        x_list = [x[:, 0, 0].view(x.size(0), -1)]
        for i in range(int(log2(L)) - 1, -1, -1):
            x_list.append(x[:, 2**i :: 2 ** (i + 1), 2**i :: 2 ** (i + 1)].reshape(x.size(0), -1))
            x_list.append(x[:, 2**i :: 2 ** (i + 1), 0 :: 2 ** (i + 1)].reshape(x.size(0), -1))
            x_list.append(x[:, 0 :: 2 ** (i + 1), 2**i :: 2 ** (i + 1)].reshape(x.size(0), -1))
        return torch.cat(x_list, dim=1)

    @staticmethod
    def unfold(x: torch.Tensor, L: int) -> torch.Tensor:
        x_out = torch.zeros_like(x).view(x.size(0), L, L)
        x_out[:, 0, 0] += x[:, 0]
        index = 1
        length = 1
        l = 1
        for i in range(int(log2(L)) - 1, -1, -1):
            x_out[:, 2**i :: 2 ** (i + 1), 2**i :: 2 ** (i + 1)] = x[
                :, index : index + length
            ].reshape(x.size(0), l, l)
            index += length
            x_out[:, 2**i :: 2 ** (i + 1), 0 :: 2 ** (i + 1)] = x[
                :, index : index + length
            ].reshape(x.size(0), l, l)
            index += length
            x_out[:, 0 :: 2 ** (i + 1), 2**i :: 2 ** (i + 1)] = x[
                :, index : index + length
            ].reshape(x.size(0), l, l)
            index += length
            length *= 4
            l *= 2
        return x_out

    def forward(self, x: torch.Tensor):
        sample_fold = MADEnew2d.fold(x, self.L)
        x_hat_fold = self.net(sample_fold)
        x_hat = MADEnew2d.unfold(x_hat_fold, self.L)
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

    def forward_onfolded(self, sample_fold: torch.Tensor):
        sample_fold = sample_fold.view(sample_fold.shape[0], -1)
        x_hat = self.net(sample_fold)
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
    @torch.compile
    def sample_onfolded(self, batch_size):
        # sample is done on the folded space and then unfolded
        sample_fold = torch.zeros(
            [batch_size, self.n], dtype=default_dtype_torch, device=self.device
        )
        x_hat_0 = self.forward_onfolded(sample_fold)
        sample_fold[:, 0] = torch.bernoulli(x_hat_0[:, 0]).to(default_dtype_torch) * 2 - 1
        for index, length in self.degeneracy:
            x_hat_fold = self.forward_onfolded(sample_fold)
            sample_fold[:, index : index + length] = (
                torch.bernoulli(x_hat_fold[:, index : index + length]).to(default_dtype_torch) * 2
                - 1
            )

        # sample = MADEnew2d.unfold(sample_fold, self.L)
        if self.z2:
            # Binary random int 0/1
            flip = (
                torch.randint(
                    2, [batch_size, 1], dtype=sample_fold.dtype, device=sample_fold.device
                )
                * 2
                - 1
            )
            sample_fold *= flip

        return sample_fold, x_hat_fold

    def sample(self, batch_size):
        sample_fold, x_hat_fold = self.sample_onfolded(batch_size)
        sample = MADEnew2d.unfold(sample_fold, self.L)
        x_hat = MADEnew2d.unfold(x_hat_fold, self.L)
        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (
            1 - mask
        )
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob_onfolded(self, sample_fold):
        x_hat_fold = self.forward_onfolded(sample_fold)
        log_prob = self._log_prob(sample_fold, x_hat_fold)
        if self.z2:
            # Density estimation on inverted sample
            sample_fold_inv = -sample_fold
            x_hat_fold_inv = self.forward_onfolded(sample_fold_inv)
            log_prob_inv = self._log_prob(sample_fold_inv, x_hat_fold_inv)
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)
        return log_prob

    def log_prob(self, sample):
        sample_fold = MADEnew2d.fold(sample, self.L)
        log_prob = self.log_prob_onfolded(sample_fold)

        return log_prob


class MADEconst2d(nn.Module):
    def __init__(self, **kwargs):
        super(MADEconst2d, self).__init__()
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
        self.degeneracy = [(self.n // 2, self.n // 2)]
        assert self.L % 2 == 0, "L must be even"
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
            SparseMaskedLinear(
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
    def fold(x: torch.Tensor, L: int) -> torch.Tensor:
        # x: (batch, L, L)
        x_list = [
            x[:, 0::2, 0::2].reshape(x.size(0), -1),
            x[:, 1::2, 1::2].reshape(x.size(0), -1),
            x[:, 0::2, 1::2].reshape(x.size(0), -1),
            x[:, 1::2, 0::2].reshape(x.size(0), -1),
        ]
        return torch.cat(x_list, dim=1)

    @staticmethod
    def unfold(x: torch.Tensor, L: int) -> torch.Tensor:
        x_out = torch.zeros_like(x).view(x.size(0), L, L)
        x_out[:, 0::2, 0::2] = x[:, 0 : L**2 // 4].view(x.size(0), L // 2, L // 2)
        x_out[:, 1::2, 1::2] = x[:, L**2 // 4 : L**2 // 2].view(x.size(0), L // 2, L // 2)
        x_out[:, 0::2, 1::2] = x[:, L**2 // 2 : 3 * L**2 // 4].view(x.size(0), L // 2, L // 2)
        x_out[:, 1::2, 0::2] = x[:, 3 * L**2 // 4 : L**2].view(x.size(0), L // 2, L // 2)
        return x_out

    def forward(self, x: torch.Tensor):
        sample_fold = MADEnew2d.fold(x, self.L)
        x_hat_fold = self.net(sample_fold)
        x_hat = MADEnew2d.unfold(x_hat_fold, self.L)
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

    def forward_onfolded(self, sample_fold: torch.Tensor):
        sample_fold = sample_fold.view(sample_fold.shape[0], -1)
        x_hat = self.net(sample_fold)
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
    def sample_onfolded(self, batch_size):
        # sample is done on the folded space and then unfolded
        sample_fold = torch.zeros(
            [batch_size, self.n], dtype=default_dtype_torch, device=self.device
        )
        x_hat_0 = self.forward_onfolded(sample_fold)
        sample_fold[:, 0] = torch.bernoulli(x_hat_0[:, 0]).to(default_dtype_torch) * 2 - 1
        current_index = 1
        for index, length in self.degeneracy:
            if current_index < index:
                for i in range(current_index, index):
                    x_hat_fold = self.forward_onfolded(sample_fold)
                    sample_fold[:, i] = torch.bernoulli(x_hat_fold[:, i]).to(default_dtype_torch) * 2 - 1
            x_hat_fold = self.forward_onfolded(sample_fold)
            sample_fold[:, index : index + length] = (
                torch.bernoulli(x_hat_fold[:, index : index + length]).to(default_dtype_torch) * 2
                - 1
            )
            current_index = index + length
            
        if current_index < self.n:
            for i in range(current_index, self.n):
                x_hat_fold = self.forward_onfolded(sample_fold)
                sample_fold[:, i] = torch.bernoulli(x_hat_fold[:, i]).to(default_dtype_torch) * 2 - 1

        # sample = MADEnew2d.unfold(sample_fold, self.L)
        if self.z2:
            # Binary random int 0/1
            flip = (
                torch.randint(
                    2, [batch_size, 1], dtype=sample_fold.dtype, device=sample_fold.device
                )
                * 2
                - 1
            )
            sample_fold *= flip

        return sample_fold, x_hat_fold

    def sample(self, batch_size):
        sample_fold, x_hat_fold = self.sample_onfolded(batch_size)
        sample = MADEnew2d.unfold(sample_fold, self.L)
        x_hat = MADEnew2d.unfold(x_hat_fold, self.L)
        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.epsilon) * mask + torch.log(1 - x_hat + self.epsilon) * (
            1 - mask
        )
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob_onfolded(self, sample_fold):
        x_hat_fold = self.forward_onfolded(sample_fold)
        log_prob = self._log_prob(sample_fold, x_hat_fold)
        if self.z2:
            # Density estimation on inverted sample
            sample_fold_inv = -sample_fold
            x_hat_fold_inv = self.forward_onfolded(sample_fold_inv)
            log_prob_inv = self._log_prob(sample_fold_inv, x_hat_fold_inv)
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)
        return log_prob

    def log_prob(self, sample):
        sample_fold = MADEnew2d.fold(sample, self.L)
        log_prob = self.log_prob_onfolded(sample_fold)

        return log_prob
