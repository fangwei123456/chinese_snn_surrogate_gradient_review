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
import numpy as np
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

def bconv1d(x, weight, stride=1, dilation=1, padding=0):
    b, c, n, h = x.shape
    n, out_channels, in_channels, kernel_width_size = weight.shape
    out = x.view(b, c * n, h)
    weight = weight.view(n * out_channels, in_channels, kernel_width_size)
    out = F.conv1d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=n, padding=padding)
    return out.view(b, c, n, -1)
def time_cat(tensor_list, t_pad):
    # tensor = torch.cat(tensor_list, dim=2)
    # if t_pad > 0:
    #     tensor = tensor[ :, :,:-t_pad]
    tensor = torch.cat(tensor_list, dim=0)
    if t_pad > 0:
        tensor = tensor[:-t_pad, :, :]
    return tensor

class Blocks(nn.Module):
    def __init__(self, n_in, n_out, t_len, t_latency, recurrent=False, beta_grad=True, adapt=True, init_beta=1, init_p=1, detach_spike_grad=True, surr_grad=FastSigmoid.apply):
        super(Blocks,self).__init__()
        self._n_in = n_in 
        self._n_out = n_out 
        self._t_len = t_len  
        self._t_latency = t_latency 
        self._recurrent = recurrent 
        self._beta_grad = beta_grad 
        self._adapt = adapt
        self._detach_spike_grad = detach_spike_grad
        self._surr_grad = surr_grad
        
        self._beta = nn.Parameter(data=torch.Tensor(n_out * [init_beta]), requires_grad=beta_grad)
        self._p = nn.Parameter(data=torch.Tensor(n_out * [init_p]), requires_grad=adapt)
        self._b = nn.Parameter(data=torch.Tensor(n_out * [1.8]), requires_grad=adapt)
        
        self._t_len_block = t_latency + 1
        self._block = Block(n_out, self._t_len_block, surr_grad)
        self._n_blocks = math.ceil(t_len / self._t_len_block)
        self._t_pad = self._n_blocks * self._t_len_block - self._t_len
        # self._p_ident_base = nn.Parameter(torch.ones(n_out,self._t_len_block), requires_grad=False)
        self._p_ident_base = nn.Parameter(torch.ones(self._t_len_block,n_out), requires_grad=False)
        self._p_exp = nn.Parameter(torch.arange(1, self._t_len_block + 1).float().unsqueeze(1), requires_grad=False)
        if recurrent==True:
            self.rec_weight = nn.Parameter(torch.rand(n_out, n_out), requires_grad=recurrent)
    def get_rec_input(self, spikes):
        return torch.einsum("ij, bj...->bi...", self.rec_weight, spikes.detach() if self._detach_spike_grad else spikes)
    @property
    def beta(self):
        return torch.clamp(self._beta, min=0.001, max=0.999)
    @property
    def p(self):
        return torch.clamp(self._p.abs(), min=0, max=0.999)
    @property
    def b(self):
        return torch.clamp(self._b.abs(), min=0.001, max=1)
    def forward(self, x):#输入[t,b,n]
        if self._t_pad != 0:
            x = F.pad(x, pad=(0,0,0,0,0,self._t_pad))
        a_kernel = torch.zeros_like(x).to(x.device)[:self._t_len_block, :, :]
        v_th = torch.ones_like(x).to(x.device)[:self._t_len_block, :, :]
        spikes_list = []
        v_init = torch.zeros_like(x[0,:, :]).to(x.device)
        int_mem = torch.zeros_like(x[0,:, :]).to(x.device)
        for i in range(self._n_blocks):
            # x_slice = x[:, :,i * self._t_len_block: (i+1) * self._t_len_block].contiguous()
            # print(i * self._t_len_block,(i+1) * self._t_len_block)
            x_slice = x[i * self._t_len_block: (i+1) * self._t_len_block, :, :]
            if i > 0:
                if self._recurrent:
                    rec_current = self.get_rec_input(spikes)
                    x_slice = x_slice + rec_current
                if self._detach_spike_grad:
                    spike_mask = spikes.detach().amax(dim=0).bool()
                else:
                    spike_mask = spikes.amax(dim=0).bool()
                refac_mask = (z < spike_mask.unsqueeze(0)) * x_slice
                x_slice -= refac_mask
                v_init = int_mem * ~spike_mask
                # if self._adapt:
                #     if self._detach_spike_grad:
                #         a_at_spike = (a_kernel * spikes.detach()).sum(dim=2) + (1 / self.p)
                #     else:
                #         a_at_spike = (a_kernel * spikes).sum(dim=2) + (1 / self.p)
                #     decay_steps = (z > 1).sum(dim=2)
                #     new_a = a_at_spike * torch.pow(self.p.unsqueeze(0), decay_steps)
                #     a = (a_kernel[:, :, -1] * ~spike_mask) + (new_a * spike_mask)
                #     a_kernel = self.compute_a_kernel(a, self.p)
                #     v_th = 1 + self.b.view(1, -1, 1) * a_kernel
                if self._adapt:
                    if self._detach_spike_grad:
                        a_at_spike = (a_kernel * spikes.detach()).sum(dim=0) + (1 / self.p)
                    else:
                        a_at_spike = (a_kernel * spikes).sum(dim=0) + (1 / self.p)
                    decay_steps = (z > 1).sum(dim=0)
                    new_a = a_at_spike * torch.pow(self.p.unsqueeze(0), decay_steps)
                    a = (a_kernel[-1, :, :] * ~spike_mask) + (new_a * spike_mask)
                    a_kernel = self.compute_a_kernel(a, self.p)
                    v_th = 1 + self.b.view(1, 1, -1) * a_kernel#较耗时间
            # spikes, z, int_mem = self._block(x_slice.permute(1,2,0).contiguous(), self.beta, v_init=v_init, v_th=v_th.permute(1,2,0).contiguous())
            spikes, z, int_mem = self._block(x_slice, self.beta, v_init=v_init, v_th=v_th)
            spikes_list.append(spikes)
        return time_cat(spikes_list, self._t_pad)
    def compute_a_kernel(self, a, p):
        return torch.pow(p.unsqueeze(0) * self._p_ident_base, self._p_exp).unsqueeze(1) * a.unsqueeze(0)    
    # def compute_a_kernel(self, a, p):
    #     tmp =  torch.pow(p.unsqueeze(-1) * self._p_ident_base, self._p_exp).unsqueeze(0)
    #     return tmp * a.unsqueeze(-1)
class Block(nn.Module):
    def __init__(self, n_in, t_len, surr_grad):
        super().__init__()
        self._n_in = n_in
        self._t_len = t_len
        self._surr_grad = surr_grad
        self._beta_exp = nn.Parameter(torch.arange(t_len).flip(0).unsqueeze(0).expand(n_in, t_len).float(),requires_grad=False)
        self._phi_kernel = nn.Parameter((torch.arange(t_len) + 1).flip(0).float().view(1, 1, 1, t_len),requires_grad=False)
    
    @staticmethod
    def g(faulty_spikes):
        negate_faulty_spikes = faulty_spikes.clone().detach()
        negate_faulty_spikes[faulty_spikes == 1.0] = 0
        faulty_spikes -= negate_faulty_spikes
        return faulty_spikes
    
    def forward(self, current, beta, v_init=None, v_th=1):
        current=current.permute(1,2,0).contiguous()
        v_th = v_th.permute(1,2,0).contiguous()
        if v_init is not None:
            current[:,:,0] += beta * v_init#较耗时间
        pad_current = F.pad(current, pad=(self._t_len - 1, 0))
        beta_kernel = self.build_beta_kernel(beta)#不耗时间
        b, n, t = pad_current.shape
        n, in_channels, kernel_width_size = beta_kernel.shape
        membrane = F.conv1d(pad_current, weight=beta_kernel, bias=None, stride=1, dilation=1,groups=n, padding=0)#很耗时间
        faulty_spikes = self._surr_grad(membrane - v_th)
        pad_spikes = F.pad(faulty_spikes, pad=(self._t_len - 1 ,0)).unsqueeze(1)
        z = F.conv2d(pad_spikes, self._phi_kernel)#很耗时间
        z = z.squeeze(1).permute(2,0,1).contiguous()
        z_copy = z.clone()
        return Block.g(z), z_copy, membrane[:,:,-1].contiguous()
    def build_beta_kernel(self, beta):
        return torch.pow(beta.unsqueeze(1), self._beta_exp).unsqueeze(1)
class ALIF(nn.Module):
    def __init__(self, n_in, n_out, t_len, t_latency, recurrent=False, beta_grad=True, adapt=True, init_beta=1, init_p=1, detach_spike_grad=True, surr_grad=FastSigmoid.apply):
        super(ALIF, self).__init__()
        self._n_in = n_in  # 输入神经元数
        self._n_out = n_out  # 输出神经元数
        self._t_len = t_len  # 时间长度
        self._t_latency = t_latency  # 潜伏期
        self._recurrent = recurrent  # 是否启用递归
        self._beta_grad = beta_grad  # beta是否可训练
        self._adapt = adapt  # 是否启用自适应
        self._detach_spike_grad = detach_spike_grad  # 是否分离脉冲梯度
        self._surr_grad = surr_grad  # 代理梯度
        self._beta = nn.Parameter(data=torch.Tensor(n_out * [init_beta]), requires_grad=beta_grad)  # 衰减常数
        self._p = nn.Parameter(data=torch.Tensor(n_out * [init_p]), requires_grad=adapt)  # 自适应系数
        self._b = nn.Parameter(data=torch.Tensor(n_out * [1.8]), requires_grad=adapt)  # 自适应系数
        if recurrent:
            self.rec_weight = nn.Parameter(torch.rand(n_out, n_out), requires_grad=recurrent)  # 递归连接权重

    def get_rec_input(self, spikes):# 计算递归输入，结合当前脉冲活动和递归权重
        return torch.einsum("ij, bj...->bi...", self.rec_weight, spikes.detach() if self._detach_spike_grad else spikes)

    @property
    def beta(self):# 保证beta在合理范围内
        return torch.clamp(self._beta, min=0.001, max=0.999)

    @property
    def p(self):# 保证自适应系数p在合理范围内
        return torch.clamp(self._p.abs(), min=0, max=0.999)

    @property
    def b(self):# 保证自适应系数b在合理范围内
        return torch.clamp(self._b.abs(), min=0.001, max=1)

    def forward(self, x):# x: 输入张量，形状为 (t, b, n) 表示时间步、批量大小、神经元数
        spikes_list = []  # 存储脉冲

        # 初始化脉冲输出
        spikes = torch.zeros_like(x).to(x.device)[0, :, :]  # shape: (b, n)

        # 初始化递归输入电流和膜电位
        rec_current = torch.zeros_like(x).to(x.device)[0, :, :]  # shape: (b, n)
        mem = torch.zeros_like(x).to(x.device)[0, :, :]  # shape: (b, n)

        # 初始化绝对不应期、阈值和自适应变量
        refac_times = torch.zeros_like(x).to(x.device)[0, :, :] + self._t_latency  # shape: (b, n)
        v_th = torch.ones_like(x).to(x.device)[0, :, :]  # shape: (b, n)
        a = torch.zeros_like(x).to(x.device)[0, :, :]  # shape: (b, n)

        for t in range(x.shape[0]):  # Iterate over time steps
            stimulus_current = x[t, :, :]  # 第t个时间的输入电流，shape: (b, n)

            # 递归输入计算
            if t >= self._t_latency and self._recurrent:
                rec_current = self.get_rec_input(spikes)
                input_current = stimulus_current + rec_current
            else:
                input_current = stimulus_current

            # 绝对不应期应用
            refac_times[spikes > 0] = 0
            refac_mask = refac_times < self._t_latency
            input_current[refac_mask] = 0
            refac_times += 1

            # 更新膜电位
            new_mem = torch.einsum("bn...,n->bn...", mem, self.beta) + input_current
            spikes = self._surr_grad(new_mem - v_th)  # 计算脉冲

            if self._detach_spike_grad:
                mem = new_mem * (1 - spikes.detach())  # 更新膜电位
            else:
                mem = new_mem * (1 - spikes)

            spikes_list.append(spikes)  # 记录脉冲

            # 自适应阈值更新
            if self._adapt:
                a = self.p * a + spikes
                v_th = 1 + self.b * a

        # 返回脉冲列表堆叠结果，沿时间维度
        return torch.stack(spikes_list, dim=0)  # shape: (t, b, n)

class DynamicReshapeModule(nn.Module):
    def __init__(self, blocks):
        super(DynamicReshapeModule, self).__init__()
        self.blocks = blocks

    def forward(self, x):
        original_shape = x.shape
        T, B = original_shape[0], original_shape[1]
        if x.dim() == 5:
            C, H, W = original_shape[2], original_shape[3], original_shape[4]
            N = C * H * W
            x_reshaped = x.view(T, B, N)
        elif x.dim() == 3:
            N = original_shape[2]
            x_reshaped = x  
        x_processed = self.blocks(x_reshaped)
        if len(original_shape) == 5:
            x_output = x_processed.view(T, B, C, H, W)
        elif len(original_shape) == 3:
            x_output = x_processed

        return x_output
    
class CIFAR10Net(nn.Module):
    def __init__(self, channels, T, latency = 1,type='blockalif'):#or alif
        super().__init__()
        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                if i==0:
                    conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                    conv.append(layer.BatchNorm2d(channels))
                    # 添加Blocks实例，设置参数
                    if type=='blockalif':
                        conv.append(DynamicReshapeModule(Blocks(n_in=channels*32*32, n_out=channels*32*32, t_len=T, t_latency=latency, recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99)))
                    elif type=='alif':
                        conv.append(DynamicReshapeModule(ALIF(n_in=channels*32*32, n_out=channels*32*32, t_len=T, t_latency=latency, recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99)))
                elif i==1:
                    conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                    conv.append(layer.BatchNorm2d(channels))
                    # 添加Blocks实例，设置参数
                    if type=='blockalif':
                        conv.append(DynamicReshapeModule(Blocks(n_in=channels*32*32, n_out=channels*16*16, t_len=T, t_latency=latency, recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99)))
                    elif type=='alif':
                        conv.append(DynamicReshapeModule(ALIF(n_in=channels*32*32, n_out=channels*16*16, t_len=T, t_latency=latency, recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99)))

            conv.append(layer.AvgPool2d(2, 2))

        self.conv = nn.Sequential(*conv)

        if type=='blockalif':
            self.fc = nn.Sequential(
                layer.Flatten(),
                layer.Linear(channels * 8 * 8, channels * 8 * 8 // 4),
                DynamicReshapeModule(Blocks(n_in=channels * 8 * 8 // 4, n_out= channels * 8 * 8 // 4, t_len=T, t_latency=latency, recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99)),
                layer.Linear(channels * 8 * 8 // 4, 10),
            )
        elif type=='alif':
            self.fc = nn.Sequential(
                layer.Flatten(),
                layer.Linear(channels * 8 * 8, channels * 8 * 8 // 4),
                DynamicReshapeModule(ALIF(n_in=channels * 8 * 8 // 4, n_out= channels * 8 * 8 // 4, t_len=T, t_latency=latency, recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99)),
                layer.Linear(channels * 8 * 8 // 4, 10),
            )

        functional.set_step_mode(self, 'm')

    def forward(self, x_seq: torch.Tensor):
        return self.fc(self.conv(x_seq))
    
from torch import Tensor
from typing import Tuple
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



def main():
    # channels = 128, params = 17542026
    '''
    python blockalif.py -data-dir ../data -amp -opt sgd -channels 128 -epochs 256 -out-dir ./logs_blockalif -T 4 -type blockalif


    python blockalif.py -data-dir /datasets/CIFAR10 -opt sgd -channels 128 -epochs 256 -out-dir ./templog

    python blockalif.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -sop -resume /home/wfang/chinese_review/cf10/logs/pt/logs/blockalifatan_la1_e256_b128_sgd_lr0.1_c128_amp/checkpoint_max.pth


    '''
    parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=256, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs_blockalif', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-T', default=4, type=int)
    parser.add_argument('-prefix', default='blockalif', type=str)
    parser.add_argument('-type', default='blockalif', type=str)
    parser.add_argument('-sop', action='store_true')

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

    net = CIFAR10Net(args.channels,args.T,type=args.type)
    net.to(args.device)

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

    if args.sop:
        import energy
        energy.get_sops_over_test_set(net, test_data_loader, args)

        exit()


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
            # print("train one data")
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                y = net(img).mean(0)
                loss = F.cross_entropy(y, label, label_smoothing=0.1)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            train_acc += (y.argmax(1) == label.argmax(1)).float().sum().item()

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