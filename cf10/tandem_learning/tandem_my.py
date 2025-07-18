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

from lib.snn import LinearBN1d_if, ConvBN2d_if
from lib.functional import InputDuplicate


class CifarNetIF(nn.Module):

    def __init__(self, neuronParam={
        'neuronType': 'IF',
        'vthr': 1,
    }, Tsim=4, channels=128):
        super(CifarNetIF, self).__init__()
        self.T = Tsim
        self.neuronParam = neuronParam
        self.conv1 = ConvBN2d_if(3, channels, 3, stride=1, padding=1, neuronParam=self.neuronParam)
        self.conv2 = ConvBN2d_if(channels, channels, 3, stride=1, padding=1, neuronParam=self.neuronParam)
        self.conv3 = ConvBN2d_if(channels, channels, 3, stride=1, padding=1, neuronParam=self.neuronParam)

        self.pool4 = nn.AvgPool2d(2, stride=2)

        self.conv5 = ConvBN2d_if(channels, channels, 3, stride=1, padding=1, neuronParam=self.neuronParam)
        self.conv6 = ConvBN2d_if(channels, channels, 3, stride=1, padding=1, neuronParam=self.neuronParam)
        self.conv7 = ConvBN2d_if(channels, channels, 3, stride=1, padding=1, neuronParam=self.neuronParam)

        self.pool8 = nn.AvgPool2d(2, stride=2)

        self.fc9 = LinearBN1d_if(8 * 8 * channels, 8 * 8 * channels // 4, neuronParam=self.neuronParam)
        self.fc10 = nn.Linear(8 * 8 * channels // 4, 10)

    @property
    def conv(self):
        return nn.Sequential(self.conv1, self.conv2, self.conv3, self.pool4, self.conv5, self.conv6, self.conv7, self.pool8)

    @property
    def fc(self):
        return nn.Sequential(self.fc9, self.fc10)

    @property
    def conv_fc(self):
        return nn.Sequential(self.conv1, self.conv2, self.conv3, self.pool4, self.conv5, self.conv6, self.conv7, self.pool8, self.fc9, self.fc10)
    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x_spike, x = InputDuplicate.apply(x, self.T)
        x_spike = x_spike.view(-1, self.T, 3, 32, 32)
        x = x.view(-1, 3, 32, 32)

        N = x_spike.shape[0]
        T = x_spike.shape[1]
        # Conv Layer
        x_spike, x = self.conv1(x_spike, x)
        x_spike, x = self.conv2(x_spike, x)
        x_spike, x = self.conv3(x_spike, x)

        x_spike = self.pool4(x_spike.flatten(0, 1)).unflatten(0, (N, T))
        x = self.pool4(x)

        x_spike, x = self.conv5(x_spike, x)
        x_spike, x = self.conv6(x_spike, x)
        x_spike, x = self.conv7(x_spike, x)

        x_spike = self.pool8(x_spike.flatten(0, 1)).unflatten(0, (N, T))
        x = self.pool8(x)

        # FC Layers
        x = x.view(x.size(0), -1)
        x_spike = x_spike.view(x_spike.size(0), self.T, -1)

        x_spike, x = self.fc9(x_spike, x)
        # x = self.fc10(x)  # x.shape = [N, C]
        x = self.fc10(x_spike).mean(1)  # 测sop时用
        return x


def record_input_shape_hook(module, input, output):
    # N, T, C, H, W

    module.input_shape = input[0][:, 0].shape
    module.output_shape = output[0][:, 0].shape


def set_input_shape(net: nn.Module, x: torch.Tensor):
    '''

    给模块设置钩子，然后给与输入，记录下突触层的输入shape，便于之后计算flop

    '''
    hds = []
    for m in net.conv_fc:
        if isinstance(m, (ConvBN2d_if, LinearBN1d_if, nn.Linear, nn.AvgPool2d, nn.AvgPool1d)):
            hds.append(m.register_forward_hook(record_input_shape_hook))

    with torch.no_grad():
        net(x)
        functional.reset_net(net)

    for h in hds:
        h.remove()

import numpy as np
# modified from https://github.com/iCGY96/syops-counter/blob/4232f771a7499d1a85ccd760c13faf394644f456/syops/ops.py
def conv_syops_counter(conv_module, input_shape, output_shape):

    batch_size = input_shape[0]
    output_dims = list(output_shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_syops = out_channels * active_elements_count

    overall_syops = overall_conv_syops + bias_syops

    return overall_syops


def linear_syops_counter(module, input_shape, output_shape):
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output_shape[-1]
    bias_syops = output_last_dim
    return int(np.prod(input_shape) * output_last_dim + bias_syops)


def set_flops(net: CifarNetIF):

    net.conv1.flops = 114688
    net.conv2.flops = 4722688
    net.conv3.flops = 4722688
    net.conv5.flops = 2361344
    net.conv6.flops = 2361344
    net.conv7.flops = 2361344
    net.fc9.flops = 16779264
    net.fc10.flops = 20490
    return
    for i in range(len(net.conv)):
        m = net.conv[i]

        if isinstance(m, ConvBN2d_if):
            m = net.conv[i]
            if isinstance(net.conv[i - 1], (nn.AvgPool1d, nn.AvgPool2d)):
                # 假定最大池化和卷积已经合并了
                m.flops = conv_syops_counter(m.conv2d, net.conv[i - 1].input_shape, m.output_shape)
            else:
                m.flops = conv_syops_counter(m.conv2d, m.input_shape, m.output_shape)
            print(m, 'flops=', m.flops)


    for m in net.fc:
        if isinstance(m, nn.Linear):
            m.flops = linear_syops_counter(m, m.input_shape, m.output_shape)
        elif isinstance(m, LinearBN1d_if):
            m.flops = linear_syops_counter(m.linear, m.input_shape, m.output_shape)

        print(m, 'flops=', m.flops)


def record_spike_hook(module, input):
    with torch.no_grad():
        module.in_spikes += input[0].sum()
        module.in_spikes_numel += input[0].numel()


def set_record_spike(net: nn.Module):
    hds = []
    for i, m in enumerate(net.conv_fc):
        if isinstance(m, (ConvBN2d_if, LinearBN1d_if, nn.Linear, nn.AvgPool2d)):
            if i >= 1:
                m.in_spikes = 0
                m.in_spikes_numel = 0
                hds.append(m.register_forward_pre_hook(record_spike_hook))
    return hds



def get_sops(net: CifarNetIF):

    # 首先将in_spikes统一到[N, -1]的shape
    in_spikes_list = []
    in_spikes_numel_list = []
    flops_list = []
    with torch.no_grad():
        for i in range(1, len(net.conv)):
            # conv[0]是直接输入图片
            m = net.conv[i]
            if isinstance(m, ConvBN2d_if):
                if isinstance(net.conv[i-1], (nn.AvgPool1d, nn.AvgPool2d)):
                    # 卷积层和平均池化层可以合并，因而sop要按照平均池化的输入来
                    in_spikes = net.conv[i-1].in_spikes
                    in_spikes_numel = net.conv[i-1].in_spikes_numel
                else:
                    in_spikes = m.in_spikes
                    in_spikes_numel = m.in_spikes_numel


                in_spikes_list.append(in_spikes)
                in_spikes_numel_list.append(in_spikes_numel)
                flops_list.append(m.flops)

        for m in net.fc:
            in_spikes_list.append(m.in_spikes)
            in_spikes_numel_list.append(m.in_spikes_numel)
            flops_list.append(m.flops)


    sops = 0
    for i in range(len(in_spikes_list)):
        # s.shape = [N, -1]
        flops = flops_list[i]
        fr = in_spikes_list[i] / in_spikes_numel_list[i]
        sops += (fr * flops).sum().item()


    return sops



def get_sops_over_test_set(net: nn.Module, test_data_loader, args):
    net.eval()
    with torch.no_grad():

        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            T = net.T
            net.T = 1
            set_input_shape(net, img[0:1])  # 获取每一层输入输出的shape
            net.T = T
            set_flops(net)  # 根据shape计算flop
            functional.reset_net(net)
            break

        set_record_spike(net)
        numel = 0
        test_acc = 0.
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            y = net(img)
            numel += label.numel()
            test_acc += (y.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
        print(test_acc / numel)
        print(f'flops={net.conv[0].flops / 1e6:.3f}, sops={get_sops(net) * args.T / 1e6:.3f}')

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


def params(net: nn.Module):
    n = 0
    for p in net.parameters():
        n += p.numel()

    return n


def main():
    # channels = 128, params = 17542026
    '''
    python tandem_my.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256

    python tandem_my.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -device cuda:1 -lr 0.001

    python tandem_my.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -lr 0.001 -sop -resume /home/wfang/chinese_review/cf10/tandem_learning/logs/pt/logs/tandem_T_4_e256_b128_sgd_lr0.001_c128_amp/checkpoint_max.pth



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
    parser.add_argument('-prefix', default='tandem', type=str)
    parser.add_argument('-sop', action='store_true')

    args = parser.parse_args()
    print(args)

    mixup_transforms = []
    mixup_transforms.append(RandomMixup(10, p=1.0, alpha=0.2))
    mixup_transforms.append(RandomCutmix(10, p=1.0, alpha=1.))
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    transform_train = ClassificationPresetTrain(mean=(0.4914, 0.4822, 0.4465),
                                                std=(0.2023, 0.1994, 0.2010),
                                                interpolation=InterpolationMode('bilinear'),
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

    net = CifarNetIF(Tsim=args.T, channels=args.channels)
    net.to(args.device)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
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
        get_sops_over_test_set(net, test_data_loader, args)

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
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                y = net(img)
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
                y = net(img)
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
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
