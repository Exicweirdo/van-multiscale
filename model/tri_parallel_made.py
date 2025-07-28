import torch
from torch import nn
from numpy import log, log2

from .special_layer import SparseMaskedLinear, ResBlock, ChannelLinear
from utils import default_dtype_torch


class TriParallelMade(nn.Module):
    def __init__(self, **kwargs):
        super(TriParallelMade, self).__init__()
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
        self.degeneracy = []
        assert log2(self.L) % 1 == 0, "L must be power of 2"
        
        for i in range(0, int(log2(self.L))):
            self.degeneracy.extend([(2**(2*i), 2**(2*i)), (2**(2*i)*2, 2**(2*i)), (2**(2*i)*3, 2**(2*i))])
        print(self.degeneracy)
        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer("x_hat_mask", torch.ones([self.L**2]))
            self.x_hat_mask[0] = 0
            self.register_buffer("x_hat_bias", torch.zeros([self.L**2]))
            self.x_hat_bias[0] = 0.5

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
        sample_fold = TriParallelMade.fold(x, self.L)
        x_hat = self.forward_onfolded(sample_fold)
        x_hat = TriParallelMade.unfold(x_hat, self.L)
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

    def partial_sample_onfolded(self, batch_size, sample_init, start=0):
        # sample is done on the folded space and then unfolded

        sample_fold = sample_init.detach().clone()
        x_hat_0 = self.forward_onfolded(sample_fold)
        if start == 0:
            sample_fold[:, 0] = torch.bernoulli(x_hat_0[:, 0]).to(default_dtype_torch) * 2 - 1

        for index, length in self.degeneracy:
            if start >= index + length:
                continue
            sampling_start_index = max(start, index)
            x_hat_fold = self.forward_onfolded(sample_fold)
            sample_fold[:, sampling_start_index : index + length] = (
                torch.bernoulli(x_hat_fold[:, sampling_start_index : index + length]).to(
                    default_dtype_torch
                )
                * 2
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
        sample = TriParallelMade.unfold(sample_fold, self.L)
        x_hat = TriParallelMade.unfold(x_hat_fold, self.L)
        return sample, x_hat

    def partial_sample(self, batch_size, sample_init=None, start=0):
        sample_init_fold = torch.zeros(
            [batch_size, self.n], dtype=default_dtype_torch, device=self.device
        )
        if sample_init is not None:
            sample_init_fold = TriParallelMade.fold(sample_init, self.L)
            
        sample_fold, x_hat_fold = self.partial_sample_onfolded(batch_size, sample_init_fold, start)
        sample = TriParallelMade.unfold(sample_fold, self.L)
        x_hat = TriParallelMade.unfold(x_hat_fold, self.L)
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
        with torch.no_grad():
            sample_fold = TriParallelMade.fold(sample, self.L)
        log_prob = self.log_prob_onfolded(sample_fold)

        return log_prob
