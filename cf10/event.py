# python event.py -backend cuda -data-dir /datasets/CIFAR10 -opt sgd -T 12 -b 64 -tau_m 7 -tau_s 4 -tau_grad 3.5 -lr 0.0001 -desired_count 10 -undesired_count 1

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime

from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from torch.nn.utils import clip_grad_norm_
from torch.utils.cpp_extension import load

try:
    neuron_cuda = load(name="neuron_cuda", sources=["event/neuron_cuda.cpp", 'event/neuron_cuda_kernel.cu'],
                    verbose=True)
except:
    print('Cannot load cuda neuron kernel.')

_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if torch.__version__ < "1.11.0":
    cpp_wrapper = load(name="cpp_wrapper", sources=["layers/cpp_wrapper.cpp"], verbose=True)
    conv_backward_input = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        cpp_wrapper.cudnn_convolution_backward_input(input.shape, grad_output, weight, padding, stride, dilation, groups,
                                                     cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
    conv_backward_weight = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        cpp_wrapper.cudnn_convolution_backward_weight(weight.shape, grad_output, input, padding, stride, dilation, groups, 
                                                      cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
else:
    bias_sizes, output_padding = [0, 0, 0, 0], [0, 0]
    transposed = False
    conv_backward_input = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        torch.ops.aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, 
                                            transposed, output_padding, groups, [True, False, False])[0]
    conv_backward_weight = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        torch.ops.aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, 
                                            transposed, output_padding, groups, [False, True, False])[1]



class smoothFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, target_spike):
        T = inputs.shape[0]
        t_start = T * 2 // 3
        target_spike = (T-t_start) / T * inputs[0].numel()
        outputs = inputs
        if (inputs >= 0).all():
            # num_spike = torch.sum(inputs[t_start:], dim=[i for i in range(len(inputs.shape)) if i != 2], keepdim=True) + 1e-5
            num_spike = torch.sum(inputs[t_start:]) + 1e-5
            outputs = inputs / num_spike * target_spike
        # ctx.save_for_backward(target_spike)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        '''
        target_spike = ctx.saved_tensors[0]
        sum0 = torch.sum(grad)
        grad = grad / (target_spike + 1e-5)
        grad = grad / torch.sum(grad) * sum0
        '''
        return grad, None


def bn_forward(weight, norm_weight, norm_bias):
    C = weight.shape[0]
    # print(weight.shape)
    mean, var = torch.mean(weight.reshape(C, -1), dim=1), torch.std(weight.reshape(C, -1), dim=1) ** 2
    shape = (-1, 1, 1, 1) if len(weight.shape) == 4 else (-1, 1)
    mean, var, norm_weight, norm_bias = [x.reshape(*shape) for x in [mean, var, norm_weight, norm_bias]]
    weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias
    return weight_


@torch.jit.script
def neuron_forward_py(in_I, threshold, theta_m, theta_s, theta_grad, is_forward_leaky, is_grad_exp):
    # syn_m & syn_s: (1-theta_m)^t & (1-theta_s)^t in eps(t)
    # syn_grad: (1-theta_grad)^t in backward
    u_last = torch.zeros_like(in_I[0])
    th_shape = torch.ones_like(torch.tensor(u_last.shape))
    th_shape: list[int] = th_shape.tolist()
    th_shape[1] = -1
    syn_m, syn_s, syn_grad = torch.zeros_like(in_I[0]), torch.zeros_like(in_I[0]), torch.zeros_like(in_I[0])
    delta_u, delta_u_t, outputs = torch.zeros_like(in_I), torch.zeros_like(in_I), torch.zeros_like(in_I)
    T = in_I.shape[0]
    for t in range(T):
        syn_m = (syn_m + in_I[t]) * (1 - theta_m)
        syn_s = (syn_s + in_I[t]) * (1 - theta_s)
        syn_grad = (syn_grad + in_I[t]) * (1 - theta_grad)

        if not is_forward_leaky:
            delta_u_t[t] = syn_grad
            u = u_last + delta_u_t[t]
            delta_u[t] = delta_u_t[t]
        else:
            u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)
            delta_u[t] = u - u_last
            delta_u_t[t] = syn_grad if is_grad_exp else delta_u[t]

        out = (u >= threshold.view(th_shape)).to(u)
        u_last = u * (1 - out)

        syn_m = syn_m * (1 - out)
        syn_s = syn_s * (1 - out)
        syn_grad = syn_grad * (1 - out)
        outputs[t] = out

    return delta_u, delta_u_t, outputs


@torch.jit.script
def neuron_backward_py(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv):
    T = grad_delta.shape[0]

    grad_in_, grad_w_ = torch.zeros_like(outputs), torch.zeros_like(outputs)
    partial_u_grad_w, partial_u_grad_t = torch.zeros_like(outputs[0]), torch.zeros_like(outputs[0])
    delta_t = torch.zeros(outputs.shape[1:], device=outputs.device, dtype=torch.long)
    spiked = torch.zeros_like(outputs[0])

    for t in range(T - 1, -1, -1):
        out = outputs[t]
        spiked += (1 - spiked) * out

        partial_u = torch.clamp(-1 / delta_u[t], -4, 0)
        partial_u_t = torch.clamp(-1 / delta_u_t[t], -max_dudt_inv, 0)
        # current time is t_m
        partial_u_grad_w = partial_u_grad_w * (1 - out) + grad_delta[t] * partial_u * out
        partial_u_grad_t = partial_u_grad_t * (1 - out) + grad_delta[t] * partial_u_t * out

        delta_t = (delta_t + 1) * (1 - out).long()
        grad_in_[t] = partial_u_grad_t * partial_a[delta_t] * spiked.to(partial_a)
        grad_w_[t] = partial_u_grad_w * syn_a[delta_t] * spiked.to(syn_a)

    return grad_in_, grad_w_


def neuron_forward(in_I, threshold, neuron_config):
    theta_m, theta_s, theta_grad = neuron_config
    theta_m, theta_s, theta_grad = torch.tensor((theta_m, theta_s, theta_grad)).to(in_I)
    assert (theta_m != theta_s)
    is_grad_exp = torch.tensor(args.gradient_type == 'exponential')
    is_forward_leaky = torch.tensor(False)
    if args.backend == 'python':
        return neuron_forward_py(in_I, threshold, theta_m, theta_s, theta_grad, is_forward_leaky, is_grad_exp)
    elif args.backend == 'cuda':
        theta_m, theta_s, theta_grad = neuron_config
        return neuron_cuda.forward(in_I, threshold, theta_m, theta_s, theta_grad, is_forward_leaky, is_grad_exp)
    else:
        raise Exception('Unrecognized computation backend.')


def neuron_backward(grad_delta, outputs, delta_u, delta_u_t):
    syn_a_, partial_a_ = syn_a.to(outputs), -delta_syn_a.to(outputs)
    max_dudt_inv = torch.tensor(123456789.)
    if args.backend == 'python':
        return neuron_backward_py(grad_delta, outputs, delta_u, delta_u_t, syn_a_, partial_a_, max_dudt_inv)
    elif args.backend == 'cuda':
        max_dudt_inv = max_dudt_inv.item()
        grad_delta = grad_delta.contiguous()
        return neuron_cuda.backward(grad_delta, outputs, delta_u, delta_u_t, syn_a_, partial_a_, max_dudt_inv)
    else:
        raise Exception('Unrecognized computation backend.')


class SpikeLoss(torch.nn.Module):
    """
    This class defines different spike based loss modules that can be used to optimize the SNN.
    """

    def __init__(self):
        super(SpikeLoss, self).__init__()

    def spike_count(self, output, target):
        # shape of output: T * N * C
        delta = loss_count.apply(output, target)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_count_plus(self, output, target):
        return loss_count_plus.apply(output, target)


class loss_count(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target):
        desired_count = args.desired_count
        undesired_count = args.undesired_count
        T = output.shape[0]
        out_count = torch.sum(output, dim=0)
        out_count[(target == desired_count) & (
            out_count > desired_count)] = desired_count
        out_count[(target == undesired_count) & (
            out_count < undesired_count)] = undesired_count

        delta = (out_count - target) / T
        delta = delta.unsqueeze_(0).repeat(T, 1, 1)
        ctx.save_for_backward(output, out_count, target)
        return delta

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        desired_count = args.desired_count
        output, out_count, target = ctx.saved_tensors
        T, N, C = output.shape
        desired_count = args.desired_count
        ratio = (torch.sum(out_count[target != desired_count]) + 1e-5) / \
            (torch.sum(out_count[target == desired_count]) + 1e-5)
        mask = (target == desired_count).unsqueeze(0).repeat(T,1,1)
        grad = grad * output
        grad[mask] = grad[mask] * max(ratio / 10, 1)
        return -grad, None


class loss_count_plus(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target):
        desired_count = args.desired_count
        undesired_count = args.undesired_count
        T = output.shape[0]
        out_count = torch.sum(output, dim=0)
        out_count[(target == desired_count) & (
            out_count > desired_count)] = desired_count
        out_count[(target == undesired_count) & (
            out_count < undesired_count)] = undesired_count

        delta = (out_count - target) / T
        ctx.save_for_backward(output, out_count, target, delta)
        delta = delta.unsqueeze_(0).repeat(T, 1, 1)
        return 1 / 2 * torch.sum(delta ** 2)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        (output, out_count, target, delta) = ctx.saved_tensors
        T, N, C = output.shape
        delta = delta.reshape(target.shape)
        desired_count = args.desired_count
        ratio = (torch.sum(out_count[target != desired_count]) + 1e-5) / \
            (torch.sum(out_count[target == desired_count]) + 1e-5)

        mask = target == desired_count
        delta[mask] = delta[mask] * max(ratio / 10, 1)
        out_count_inv = 1 / out_count
        out_count_inv[out_count == 0] = 0
        delta = delta * out_count_inv
        delta = delta.unsqueeze_(0).repeat(T, 1, 1) * output
        
        sign = -1
        return sign * delta, None


class ConvLayer(nn.Conv2d):
    def __init__(self, config, name=None, groups=1):
        self.name = name
        threshold = 1
        self.type = 'conv'
        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size']

        padding = config['padding'] if 'padding' in config else 0
        stride = config['stride'] if 'stride' in config else 1
        dilation = config['dilation'] if 'dilation' in config else 1

        readConfig = lambda x: x if isinstance(x, (tuple, list)) else (x, x)
        self.kernel = readConfig(kernel_size)
        self.stride = readConfig(stride)
        self.padding = readConfig(padding)
        self.dilation = readConfig(dilation)

        super(ConvLayer, self).__init__(in_features, out_features, self.kernel, self.stride, self.padding,
                                        self.dilation, groups, bias=False)
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        self.norm_weight = torch.nn.Parameter(torch.ones(out_features, 1, 1, 1, device='cuda'))
        self.norm_bias = torch.nn.Parameter(torch.zeros(out_features, 1, 1, 1, device='cuda'))
        self.register_buffer('threshold', torch.ones(out_features, 1, 1, 1) * threshold)

        print('conv')
        print(f'Shape of weight is {list(self.weight.shape)}')  # Cout * Cin * Hk * Wk
        print(f'stride = {self.stride}, padding = {self.padding}, dilation = {self.dilation}, groups = {self.groups}')
        print("-----------------------------------------")

    def forward_pass(self, x):
        theta_m = 1 / args.tau_m
        theta_s = 1 / args.tau_s
        theta_grad = 1 / args.tau_grad if args.gradient_type == 'exponential' else -123456789  # instead of None
        
        normed_weight = bn_forward(self.weight, self.norm_weight, self.norm_bias)
        y = ConvFunc.apply(x, normed_weight, self.threshold, (self.bias, self.stride, self.padding, self.dilation, self.groups), (theta_m, theta_s, theta_grad))
        return y

    def forward(self, x):
        with torch.no_grad():
            if True:
                # print(x.shape, torch.min(self.norm_weight.data))
                assert(torch.min(self.norm_weight.data) > 0)
                # self.threshold.data /= self.norm_weight.data
                self.threshold = self.threshold / self.norm_weight.data
                self.norm_bias.data /= self.norm_weight.data
                self.norm_weight.data = torch.ones_like(self.norm_weight.data)
            self.weight_clipper()

        y = self.forward_pass(x)
        return y

    def forward_pass(self, x):
        theta_m = 1 / args.tau_m
        theta_s = 1 / args.tau_s
        theta_grad = 1 / args.tau_grad if args.gradient_type == 'exponential' else -123456789  # instead of None
        
        normed_weight = bn_forward(self.weight, self.norm_weight, self.norm_bias)
        y = ConvFunc.apply(x, normed_weight, self.threshold, (self.bias, self.stride, self.padding, self.dilation, self.groups), (theta_m, theta_s, theta_grad))
        return y

    def weight_clipper(self):
        self.weight.data = self.weight.data.clamp(-4, 4)
        # self.threshold.data = self.threshold.data.clamp(self.th0 / 5, self.th0 * 5)
        # self.norm_weight.data = self.norm_weight.data.clamp(self.normw0 / 5, self.normw0 * 5)


class ConvFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, threshold, conv_config, neuron_config):
        # input.shape: T * n_batch * C_in * H_in * W_in
        bias, stride, padding, dilation, groups = conv_config
        T, n_batch, C, H, W = inputs.shape

        in_I = F.conv2d(inputs.reshape(T * n_batch, C, H, W), weight, bias, stride, padding, dilation, groups)
        _, C, H, W = in_I.shape
        in_I = in_I.reshape(T, n_batch, C, H, W)

        delta_u, delta_u_t, outputs = neuron_forward(in_I, threshold, neuron_config)

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, threshold)
        ctx.conv_config = conv_config

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * C * H * W
        (delta_u, delta_u_t, inputs, outputs, weight, threshold) = ctx.saved_tensors
        bias, stride, padding, dilation, groups = ctx.conv_config
        grad_delta *= outputs
        # sum_next = grad_delta.sum().item()
        # print("Max of dLdt: ", abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)

        T, n_batch, C, H, W = grad_delta.shape
        inputs = inputs.reshape(T * n_batch, *inputs.shape[2:])
        grad_in_, grad_w_ = map(lambda x: x.reshape(T * n_batch, C, H, W), [grad_in_, grad_w_])
        
        grad_input = conv_backward_input(grad_in_, inputs, weight, padding, stride, dilation, groups) * inputs
        grad_weight = conv_backward_weight(grad_w_, inputs, weight, padding, stride, dilation, groups)

        # sum_last = grad_input.sum().item()
        # print(f'sum_next = {sum_next}, sum_last = {sum_last}')
        # assert(abs(sum_next - sum_last) < 1)
        
        return grad_input.reshape(T, n_batch, *inputs.shape[1:]), grad_weight, None, None, None


class LinearLayer(nn.Linear):
    def __init__(self, config, name=None):
        # extract information for kernel and inChannels
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        threshold = 1
        self.name = name
        self.type = 'linear'
        # self.in_shape = in_shape
        # self.out_shape = [out_features, 1, 1]

        if type(in_features) == int:
            n_inputs = in_features
        else:
            raise Exception('inFeatures should not be more than 1 dimension. It was: {}'.format(in_features.shape))
        if type(out_features) == int:
            n_outputs = out_features
        else:
            raise Exception('outFeatures should not be more than 1 dimension. It was: {}'.format(out_features.shape))

        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        self.norm_weight = torch.nn.Parameter(torch.ones(out_features,1, device='cuda'))
        self.norm_bias = torch.nn.Parameter(torch.zeros(out_features,1, device='cuda'))
        self.register_buffer('threshold', torch.ones(out_features, 1) * threshold)

        print("linear")
        print(self.name)
        # print(self.in_shape)
        # print(self.out_shape)
        print(f'Shape of weight is {list(self.weight.shape)}')
        print("-----------------------------------------")

    def forward_pass(self, x, labels=None):
        ndim = len(x.shape)
        assert (ndim == 3 or ndim == 5)
        if ndim == 5:
            T, n_batch, C, H, W = x.shape
            x = x.view(T, n_batch, C * H * W)
        theta_m = 1 / args.tau_m
        theta_s = 1 / args.tau_s
        theta_grad = 1 / args.tau_grad if args.gradient_type == 'exponential' else -123456789  # instead of None
        
        normed_weight = bn_forward(self.weight, self.norm_weight, self.norm_bias)
        y = LinearFunc.apply(x, normed_weight, self.threshold, (theta_m, theta_s, theta_grad), labels)
        return y

    def forward(self, x, labels=None):
        with torch.no_grad():
            if True:
                assert(torch.min(self.norm_weight.data) > 0)
                # self.threshold.data /= self.norm_weight.data
                self.threshold = self.threshold / self.norm_weight.data
                self.norm_bias.data /= self.norm_weight.data
                self.norm_weight.data = torch.ones_like(self.norm_weight.data)
            self.weight_clipper()

        y = self.forward_pass(x, labels)
        return y

    def weight_clipper(self):
        self.weight.data = self.weight.data.clamp(-4, 4)
        # self.threshold.data = self.threshold.data.clamp(self.th0 / 5, self.th0 * 5)
        # self.norm_weight.data = self.norm_weight.data.clamp(self.normw0 / 5, self.normw0 * 5)


class LinearFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, threshold, config, labels):
        #input.shape: T * n_batch * N_in
        in_I = torch.matmul(inputs, weight.t())

        T, n_batch, N = in_I.shape
        theta_m, theta_s, theta_grad = config
        assert (theta_m != theta_s)
        delta_u, delta_u_t, outputs = neuron_forward(in_I, threshold, config)

        if labels is not None:
            glv.outputs_raw = outputs.clone()
            i2 = torch.arange(n_batch)
            # Add supervisory signal when synaptic potential is increasing:
            is_inc = (delta_u[:, i2, labels] > 0.05).float()
            _, i1 = torch.max(is_inc * torch.arange(1, T+1, device=is_inc.device).unsqueeze(-1), dim=0)
            outputs[i1, i2, labels] = (delta_u[i1, i2, labels] != 0).to(outputs)

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, threshold)
        ctx.is_out_layer = labels != None

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * N_out

        (delta_u, delta_u_t, inputs, outputs, weight, threshold) = ctx.saved_tensors
        grad_delta *= outputs
        # sum_next = grad_delta.sum().item()
        # print("Max of dLdt: ", abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)

        grad_input = torch.matmul(grad_in_, weight) * inputs
        grad_weight = torch.sum(torch.matmul(grad_w_.transpose(1, 2), inputs), dim=0)
        
        # sum_last = grad_input.sum().item()
        # print(f'sum_next = {sum_next}, sum_last = {sum_last}')
        # assert(abs(sum_next - sum_last) < 1)

        return grad_input, grad_weight, None, None, None


class PoolLayer(nn.Module):
    def __init__(self, size, name=None):
        super(PoolLayer, self).__init__()
        self.name = name
        self.type = 'pooling'

        self.kernel = size if isinstance(size, (tuple, list)) else (size, size)
        print('pooling')
        print("-----------------------------------------")

    def forward(self, x):
        T, n_batch, C, H, W = x.shape
        x = x.reshape(T * n_batch, C, H, W)
        # x = f.max_pool2d(x, self.kernel)
        x = PoolFunc.apply(x, self.kernel)
        x = x.reshape(T, n_batch, *x.shape[1:])
        return x

    def get_parameters(self):
        return self.weight

    def weight_clipper(self):
        return


class PoolFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, kernel):
        outputs = F.avg_pool2d(inputs, kernel)
        ctx.save_for_backward(outputs, torch.tensor(inputs.shape), torch.tensor(kernel))
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        (outputs, input_shape, kernel) = ctx.saved_tensors
        kernel = kernel.tolist()
        outputs = 1 / outputs
        outputs[outputs > kernel[0] * kernel[1] + 1] = 0
        outputs /= kernel[0] * kernel[1]
        grad = F.interpolate(grad_delta * outputs, size=input_shape.tolist()[2:])
        return grad, None


class ClassificationPresetTrain:
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = []
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

from torch import Tensor
from typing import Tuple
class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = torchvision.transforms.functional.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class CIFAR10Net(nn.Module):
    def __init__(self, channels):
        super().__init__()
        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                # conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                # conv.append(layer.BatchNorm2d(channels))
                # conv.append(neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='cupy'))
                config = {'in_channels': in_channels, 'out_channels': channels, 'kernel_size': 3, 'padding': 1}
                conv.append(ConvLayer(config))

            conv.append(PoolLayer(size=2))

        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            # layer.Flatten(),
            # layer.Linear(channels * 8 * 8, channels * 8 * 8 // 4),
            # neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', backend='cupy'),
            LinearLayer(config = {'n_inputs': channels * 8 * 8, 'n_outputs': channels * 8 * 8 // 4}),
            # layer.Linear(channels * 8 * 8 // 4, 10),
            LinearLayer(config = {'n_inputs': channels * 8 * 8 // 4, 'n_outputs': 10}),
        )

    def init_param(self, spikes):
        avg_spike_init = 1.2
        from math import sqrt
        T = spikes.shape[0]
        t_start = T * 2 // 3

        for layer in list(self.conv.children()) + list(self.fc.children()):
            if isinstance(layer, (ConvLayer, LinearLayer)):
                with torch.no_grad():
                    low, high = 0.01, 100
                    while high / low >= 1.01:
                        mid = sqrt(high * low)
                        layer.norm_weight.data *= mid
                        outputs = layer.forward(spikes)
                        layer.norm_weight.data /= mid
                        n_neuron = outputs[0].numel()
                        avg_spike = torch.sum(outputs[t_start:]) / n_neuron
                        if avg_spike > avg_spike_init / T * (T - t_start):
                            high = mid
                        else:
                            low = mid
                    layer.threshold.data /= mid
                    # print(f'Average spikes per neuron = {torch.sum(outputs) / n_neuron}')
            spikes = layer(spikes)

    def forward(self, x_seq: torch.Tensor, label=None):
        x_seq = self.conv(x_seq)
        fc_num = len(list(self.fc.children()))
        for i, layer in enumerate(self.fc.children()):
            if i < fc_num - 1:
                x_seq = layer(x_seq)
            else:
                x_seq = layer(x_seq, label)
        return x_seq


def calc_delta_syn():
    global syn_a, delta_syn_a
    syn_a, delta_syn_a = (torch.zeros(args.T + 1, device=torch.device(args.device)) for _ in range(2))
    theta_m, theta_s = 1 / args.tau_m, 1 / args.tau_s
    if args.gradient_type == 'exponential':
        tau_grad = args.tau_grad
        theta_grad = 1 / tau_grad

    for t in range(args.T):
        t1 = t + 1
        syn_a[t] = ((1 - theta_m) ** t1 - (1 - theta_s) ** t1) * theta_s / (theta_s - theta_m)
        if args.gradient_type == 'exponential':
            delta_syn_a[t] = (1 - theta_grad) ** t1
        else:
            f = lambda t: ((1 - theta_m) ** t - (1 - theta_s) ** t) * theta_s / (theta_s - theta_m)
            delta_syn_a[t] = f(t1) - f(t1 - 1)


def params(net: nn.Module):
    n = 0
    for p in net.parameters():
        n += p.numel()

    return n


class GLV:
    def __init__(self):
        self.outputs_raw = None


def main():
    global args, glv
    glv = GLV()
    # channels = 128, params = 17542026
    '''
    python proxy.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256


    '''
    parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=256, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-T', default=4, type=int)
    parser.add_argument('-gradient_type', default='exponential', type=str)
    parser.add_argument('-tau_m', default=3, type=float)
    parser.add_argument('-tau_s', default=2.5, type=float)
    parser.add_argument('-tau_grad', default=2, type=float)
    parser.add_argument('-desired_count', default=5, type=int)
    parser.add_argument('-undesired_count', default=1, type=int)
    parser.add_argument('-backend', default='python', type=str)
    parser.add_argument('-prefix', default='event', type=str)


    args = parser.parse_args()
    print(args)

    mixup_transforms = []
    mixup_transforms.append(RandomMixup(10, p=1.0, alpha=0.2))
    mixup_transforms.append(RandomCutmix(10, p=1.0, alpha=1.))
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    transform_train = ClassificationPresetTrain(mean=(0.4914, 0.4822, 0.4465),
                                                  std=(0.2023, 0.1994, 0.2010), interpolation=InterpolationMode('bilinear'),
                                                  auto_augment_policy='ta_wide',
                                                  random_erase_prob=0.1)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=transform_train,
            download=True)

    test_set = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=transform_test,
            download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    out_dir = f'{args.prefix}_T_{args.T}_e{args.epochs}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}'
    if args.amp:
        out_dir += '_amp'

    out_dir = os.path.join(args.out_dir, out_dir)
    pt_dir = os.path.join(args.out_dir, 'pt', out_dir)
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    net = CIFAR10Net(args.channels)
    net.to(args.device)
    inputs = torch.stack([train_data_loader.dataset[i][0] for i in range(args.b)], dim=0).to(args.device)
    inputs = inputs.unsqueeze_(0).repeat(args.T, 1, 1, 1, 1)
    print(inputs.shape)
    net.init_param(inputs)
    calc_delta_syn()
    error = SpikeLoss().to(args.device)

    if args.T == 1:
        functional.set_backend(net, 'torch')

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.round().to(args.device).long()

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                # y = net(img).mean(0)
                # loss = F.cross_entropy(y, label, label_smoothing=0.1)
                y = net(img, label.argmax(1))
                targets = torch.ones_like(y[0]) * args.undesired_count
                for i in range(len(label)):
                    targets[i, label.argmax(1)[i]] = args.desired_count
                # print(y.shape, targets.shape)
                loss = error.spike_count_plus(y, targets)
                y = y.mean(0)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            clip_grad_norm_(net.parameters(), 1000)

            # print(glv.outputs_raw, label)
            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            train_acc += (glv.outputs_raw.mean(0).argmax(1) == label.argmax(1)).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                y = net(img).mean(0)
                loss = F.cross_entropy(y, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (y.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')



if __name__ == '__main__':
    main()