import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale=10):
        ctx.scale = scale
        ctx.save_for_backward(input)

        return input.gt(0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2
        return grad, None
class BoxCar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        grad[input <= -0.5] = 0
        grad[input > 0.5] = 0
        return grad
class MG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        lens = 0.5
        hight = 0.15
        scale = 6
        gamma = 0.5
        temp = MG.gaussian(input, mu=0., sigma=lens) * (1. + hight) - MG.gaussian(input, mu=lens,
                                                                                  sigma=scale * lens) * hight - MG.gaussian(
            input, mu=-lens, sigma=scale * lens) * hight
        return gamma * grad * temp.float()
    @staticmethod
    def gaussian(x, mu=0., sigma=.5):
        return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma
def spike(x, type):
    if type == "fast_sigmoid":
        return FastSigmoid.apply(x)
    elif type == "box_car":
        return BoxCar.apply(x)
    elif type == "mg":
        return MG.apply(x)
def bconv1d(x, weight, stride=1, dilation=1, padding=0):
    # Would be useful if PyTorch provided batched 1D convs in their library
    b, c, n, h = x.shape
    n, out_channels, in_channels, kernel_width_size = weight.shape
    out = x.view(b, c * n, h)
    weight = weight.view(n * out_channels, in_channels, kernel_width_size)
    out = F.conv1d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=n, padding=padding)
    return out.view(b, c, n, -1)
def time_cat(tensor_list, t_pad):
    tensor = torch.cat(tensor_list, dim=2)
    if t_pad > 0:
        tensor = tensor[:, :, :-t_pad]
    return tensor

class Block(nn.Module):
    def __init__(self, n_in, t_len, surr_grad):
        super().__init__()
        self._n_in = n_in
        self._t_len = t_len
        self._surr_grad = surr_grad

        self._beta_ident_base = nn.Parameter(torch.ones(n_in, t_len), requires_grad=False)
        self._beta_exp = nn.Parameter(torch.arange(t_len).flip(0).unsqueeze(0).expand(n_in, t_len).float(), requires_grad=False)
        self._phi_kernel = nn.Parameter((torch.arange(t_len) + 1).flip(0).float().view(1, 1, 1, t_len), requires_grad=False)

    @staticmethod
    def g(faulty_spikes):
        negate_faulty_spikes = faulty_spikes.clone().detach()
        negate_faulty_spikes[faulty_spikes == 1.0] = 0
        faulty_spikes -= negate_faulty_spikes

        return faulty_spikes

    def forward(self, current, beta, v_init=None, v_th=1, mode="train"):

        if v_init is not None:
            current[:, :, 0] += beta * v_init

        pad_current = F.pad(current, pad=(self._t_len - 1, 0)).unsqueeze(1)

        # compute membrane potential without reset
        beta_kernel = self.build_beta_kernel(beta)
        membrane = bconv1d(pad_current, beta_kernel)

        # map no-reset membrane potentials to output spikes
        v_th = v_th.unsqueeze(1)
        faulty_spikes = spike(membrane - v_th, self._surr_grad)

        pad_spikes = F.pad(faulty_spikes, pad=(self._t_len - 1, 0))
        z = F.conv2d(pad_spikes, self._phi_kernel)
        z_copy = z.clone().squeeze(1)

        if mode == "train":
            return Block.g(z).squeeze(1), z_copy, membrane.squeeze(1)
        elif mode == "val":
            return Block.g(z).squeeze(1), z_copy, faulty_spikes, membrane.squeeze(1)

    def build_beta_kernel(self, beta):
        beta_base = beta.unsqueeze(1).multiply(self._beta_ident_base)
        return torch.pow(beta_base, self._beta_exp).unsqueeze(1).unsqueeze(1)


# from brainbox.models import BBModel
class BaseSNN(nn.Module):
    MIN_BETA = 0.001
    MAX_BETA = 0.999
    def __init__(self, n_in, n_out, rf_len, t_len, t_latency, recurrent=True, beta_grad=True, adapt=True, init_beta=1, init_p=1, detach_spike_grad=True, surr_grad="fast_sigmoid"):
        super().__init__()
        self._n_in = n_in
        self._n_out = n_out
        self._rf_len = rf_len
        self._t_len = t_len
        self._t_latency = t_latency
        self._recurrent = recurrent
        self._beta_grad = beta_grad
        self._adapt = adapt
        self._detach_spike_grad = detach_spike_grad
        self._surr_grad = surr_grad

        self._beta = nn.Parameter(data=torch.Tensor(n_out * [init_beta]), requires_grad=beta_grad)
        # self._rf_weight = nn.Parameter(torch.rand(n_out, 1, n_in, self._rf_len), requires_grad=True)
        # self._rf_bias = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        if self._recurrent:
            self._rec_weight = nn.Parameter(torch.rand(n_out, n_out), requires_grad=recurrent)
            self.init_weight(self._rec_weight, "identity")

        self._p = nn.Parameter(data=torch.Tensor(n_out * [init_p]), requires_grad=adapt)
        self._b = nn.Parameter(data=torch.Tensor(n_out * [1.8]), requires_grad=adapt)

        # self.init_weight(self._rf_weight, "uniform", a=-1 / np.sqrt(n_in * rf_len), b=1 / np.sqrt(n_in * rf_len))

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out, "rf_len": self._rf_len, "t_len": self._t_len, "t_latency": self._t_latency, "recurrent": self._recurrent, "beta_grad": self._beta_grad, "adapt": self._adapt, "detach_spike_grad": self._detach_spike_grad, "surr_grad": self._surr_grad}

    @property
    def p(self):
        return torch.clamp(self._p.abs(), min=0, max=0.999)

    @property
    def b(self):
        return torch.clamp(self._b.abs(), min=0.001, max=1)

    @property
    def beta(self):
        return torch.clamp(self._beta, min=BaseSNN.MIN_BETA, max=BaseSNN.MAX_BETA)

    @property
    def rec_weight(self):
        return self._rec_weight

    def get_rec_input(self, spikes):
        return torch.einsum("ij, bj...->bi...", self.rec_weight, spikes.detach() if self._detach_spike_grad else spikes)

    def forward(self, x, mode="train"):
        # x: b x n x t
        # x = F.pad(x, (self._rf_len - 1, 0))
        # x = x.unsqueeze(1)  # Add channel dim
        # x = F.conv2d(x, self._rf_weight, self._rf_bias)[:, :, 0]  # Slice out height dim
        return self.process(x, mode)
    def process(self, x, mode):
        raise NotImplementedError

class Blocks(BaseSNN):
    def __init__(self, n_in, n_out, rf_len, t_len, t_latency, recurrent=True, beta_grad=True,pt=True, init_beta=1, init_p=1, detach_spike_grad=True, surr_grad="fast_sigmoid"):
        super().__init__(n_in, n_out, rf_len, t_len, t_latency, recurrent, beta_grad, adapt, init_beta, init_p, detach_spike_grad, surr_grad)

        self._t_len_block = t_latency + 1
        self._block = Block(n_out, self._t_len_block, surr_grad)
        self._n_blocks = math.ceil(t_len / self._t_len_block)
        self._t_pad = self._n_blocks * self._t_len_block - self._t_len

        self._p_ident_base = nn.Parameter(torch.ones(n_out, self._t_len_block), requires_grad=False)
        self._p_exp = nn.Parameter(torch.arange(1, self._t_len_block + 1).float(), requires_grad=False)

    def process(self, x, mode="train"):
        x_init = x
        if self._t_pad != 0:
            x = F.pad(x, pad=(0, self._t_pad))

        mem_list = []
        spikes_list = []
        z_list = []

        z = torch.zeros_like(x[:, :, self._t_len_block:])
        v_init = torch.zeros_like(x[:, :, 0]).to(x.device)
        int_mem = torch.zeros_like(x[:, :, 0]).to(x.device)

        a_kernel = torch.zeros_like(x).to(x.device)[:, :, :self._t_len_block]
        v_th = torch.ones_like(x).to(x.device)[:, :, :self._t_len_block]
        v_th_list = []

        for i in range(self._n_blocks):
            x_slice = x[:, :, i * self._t_len_block: (i+1) * self._t_len_block].clone()

            # Recurrent current and refractory mask only included after first block
            if i > 0:
                # Add recurrent current to input
                if self._recurrent:
                    rec_current = self.get_rec_input(spikes)
                    x_slice = x_slice + rec_current

                # Apply refractory mask to input
                if self._detach_spike_grad:
                    spike_mask = spikes.detach().amax(dim=2).bool()
                else:
                    spike_mask = spikes.amax(dim=2).bool()
                refac_mask = (z < spike_mask.unsqueeze(2)) * x_slice
                x_slice -= refac_mask

                # Set initial membrane potentials
                v_init = int_mem[:, :, -1] * ~spike_mask  # if spiked -> zero initial membrane potential

                # Set initial adaptive params
                if self._adapt:
                    # Get a at time of spike + spike (which is equal to 1/p to account for raising v_th by 1 next step
                    # do the math or see paper if this is not clear)
                    if self._detach_spike_grad:
                        a_at_spike = (a_kernel * spikes.detach()).sum(dim=2) + (1 / self.p)
                    else:
                        a_at_spike = (a_kernel * spikes).sum(dim=2) + (1 / self.p)
                    decay_steps = (z > 1).sum(dim=2)  # Compute number of decay steps
                    new_a = a_at_spike * torch.pow(self.p.unsqueeze(0), decay_steps)
                    a = (a_kernel[:, :, -1] * ~spike_mask) + (new_a * spike_mask)

                    # Update a for neurons that spiked
                    a_kernel = self.compute_a_kernel(a, self.p)
                    v_th = 1 + self.b.view(1, -1, 1) * a_kernel

            if mode == "train":
                spikes, z, int_mem = self._block(x_slice, self.beta, v_init=v_init, v_th=v_th, mode="train")
                spikes_list.append(spikes)
            elif mode == "val":
                spikes, z, _, int_mem = self._block(x_slice, self.beta, v_init=v_init, v_th=v_th, mode="val")
                spikes_list.append(spikes)
                mem_list.append(int_mem)
                z_list.append(z)
                v_th_list.append(v_th)

        if mode == "train":
            return time_cat(spikes_list, self._t_pad)
        elif mode == "val":
            return time_cat(spikes_list, self._t_pad), time_cat(mem_list, self._t_pad), x_init, time_cat(z_list, self._t_pad), time_cat(v_th_list, self._t_pad)

    def compute_a_kernel(self, a, p):
        # a: b x n
        # p: n
        # output: b x n x t

        return torch.pow(p.unsqueeze(-1) * self._p_ident_base, self._p_exp).unsqueeze(0) * a.unsqueeze(-1)
class SNN(BaseSNN):

    def __init__(self, n_in, n_out, rf_len, t_len, t_latency, recurrent=False, beta_grad=True, adapt=True, init_beta=1, init_p=1, detach_spike_grad=True, surr_grad="fast_sigmoid"):
        super().__init__(n_in, n_out, rf_len, t_len, t_latency, recurrent, beta_grad, adapt, init_beta, init_p, detach_spike_grad, surr_grad)

    def process(self, x, mode="train"):
        # x: b x n x t

        mem_list = []
        spikes_list = []
        spikes = torch.zeros_like(x).to(x.device)[:, :, 0]
        rec_current = torch.zeros_like(x)
        mem = torch.zeros_like(x).to(x.device)[:, :, 0]
        refac_times = torch.zeros_like(x).to(x.device)[:, :, 0] + self._t_latency

        v_th = torch.ones_like(x).to(x.device)[:, :, 0]
        a = torch.zeros_like(x).to(x.device)[:, :, 0]
        v_th_list = []

        for t in range(x.shape[2]):
            stimulus_current = x[:, :, t].clone()

            # Recurrent latency
            if t >= self._t_latency and self._recurrent:
                rec_current[:, :, t] = self.get_rec_input(spikes)
                input_current = stimulus_current + rec_current[:, :, t-self._t_latency]
            else:
                input_current = stimulus_current

            # Apply absolute refractory period
            refac_times[spikes > 0] = 0
            refac_mask = refac_times < self._t_latency
            input_current[refac_mask] = 0
            refac_times += 1

            new_mem = torch.einsum("bn...,n->bn...", mem, self.beta) + input_current
            spikes = spike(new_mem - v_th, self._surr_grad)

            mem_list.append(new_mem)
            if self._detach_spike_grad:
                mem = new_mem * (1 - spikes.detach())
            else:
                mem = new_mem * (1 - spikes)
            # new_mem -= new_mem * spikes (should be same as above?)
            spikes_list.append(spikes)

            if self._adapt:
                a = self.p * a + spikes
                v_th = 1 + self.b * a
            v_th_list.append(v_th)

        if mode == "train":
            return torch.stack(spikes_list, dim=2)
        elif mode == "val":
            v_th = torch.stack(v_th_list, dim=2)
            v_th = torch.roll(v_th, 1, dims=2)
            v_th[:, :, :1] = 1

            return torch.stack(spikes_list, dim=2), torch.stack(mem_list, dim=2), x, v_th