# python online.py -data-dir /datasets/CIFAR10 -opt sgd -device cuda:1
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

from torch import Tensor
from typing import Tuple, Callable

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
args = None


class OSR(nn.Module):
    def __init__(self, T, num_features):
        super(OSR, self).__init__()
        self.eps = 1e-5
        self.init = False
        self.shape = [1, num_features, 1, 1]
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('run_mean', torch.zeros(num_features))
        self.register_buffer('run_var', torch.ones(num_features))
        # for estimating total mean and var
        self.total_mean = 0.
        self.total_var = 0.
        self.momentum = 1 - (1 - 0.9) / T
        self.T = T

    def forward(self, x):
        eps = self.eps
        
        if self.training and self.init and isinstance(self.total_var, torch.Tensor):
            with torch.no_grad():
                mean = self.total_mean / self.T
                var = self.total_var / self.T
                if args.BN_type == 'new': var -= mean ** 2
                self.run_mean += (1 - self.momentum) * (mean - self.run_mean)
                self.run_var += (1 - self.momentum) * (var - self.run_var)
                self.total_mean = 0.
                self.total_var = 0.
        
        # count_all = torch.full((1,), x.numel() // x.size(1), dtype = x.dtype, device=x.device)
        if self.training:
            dims = [0, 2, 3]
            mean, var = x.mean(dim=dims), x.var(dim=dims)
            invstd = 1. / torch.sqrt(var + eps)
            
            with torch.no_grad():
                self.total_mean += mean
                self.total_var += var
                if args.BN_type == 'new': self.total_var += mean ** 2

            y1 = (x - mean.reshape(self.shape)) * invstd.reshape(self.shape)
            with torch.no_grad():
                scale = torch.sqrt(var + eps) / torch.sqrt(self.run_var + eps)
                if self.training and args.BN_type == 'new':
                    bound = 5.
                    scale = torch.clip(scale, 1. / bound, bound)
                shift = (mean - self.run_mean) * scale / torch.sqrt(var + eps)
            y = (y1 * scale.reshape(self.shape) + shift.reshape(self.shape)) * self.gamma.reshape(self.shape) + self.beta.reshape(self.shape)
        else:
            mean, invstd = self.run_mean, 1. / torch.sqrt(self.run_var + eps)
            y = torch.batch_norm_elemt(x, self.gamma, self.beta, mean, invstd, eps)

        self.init = False
        return y


class OnlineLIFNode(neuron.LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.dynamic_threshold = True
        self.fixed_test_threshold = True
        if self.dynamic_threshold and self.fixed_test_threshold:
            self.register_buffer('run_th', torch.ones(1))
            self.th_momentum = 1 - (1 - 0.9) / args.T
        self.init_threshold = v_threshold
        self.th_ratio = None

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            self.v = self.v.detach() * self.decay + x
        else:
            self.v = self.v.detach() * self.decay + self.v_reset * (1. - self.decay) + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor, shape=None):
        if shape is None:
            self.v = torch.zeros_like(x)
        else:
            self.v = torch.zeros(*shape, device=x.device)
        # self.v = 0.
    
    def get_decay_coef(self):
        self.decay = torch.tensor(1 - 1. / self.tau)
    
    def adjust_th(self):
        if self.dynamic_threshold:
            if not self.fixed_test_threshold or self.train():
                with torch.no_grad():
                    x = self.v
                    mean, std = torch.mean(x), torch.std(x)
                    if self.init:
                        self.th_ratio = (self.init_threshold - mean) / std
                    self.v_threshold = mean + std * self.th_ratio
                if self.fixed_test_threshold:
                    self.run_th += (1 - self.th_momentum) * (self.v_threshold - self.run_th)
            else:
                self.v_threshold = self.run_th.item()
        #print(self.v_threshold)
    def forward(self, x: torch.Tensor):
        if self.init:
            self.forward_init(x)

        self.get_decay_coef()
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        self.adjust_th()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        self.init = False
        return spike


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


class CIFAR10Net(nn.Module):
    def __init__(self, channels, T):
        super().__init__()
        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(OSR(T, channels))
                # conv.append(layer.BatchNorm2d(channels))
                # conv.append(neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='s', backend='cupy'))
                conv.append(OnlineLIFNode(tau=2., surrogate_function=surrogate.ATan(), detach_reset=True))

            conv.append(layer.AvgPool2d(2, 2))

        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(channels * 8 * 8, channels * 8 * 8 // 4),
            # neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='s', backend='cupy'),
            OnlineLIFNode(tau=2., surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Linear(channels * 8 * 8 // 4, 10),
        )

        functional.set_step_mode(self, 's')

    def forward(self, x_seq: torch.Tensor):
        return self.fc(self.conv(x_seq))

    def set_init(self):
        for layer in list(self.conv.children()) + list(self.fc.children()):
            if isinstance(layer, (neuron.BaseNode, OSR)):
                layer.init = True



def params(net: nn.Module):
    n = 0
    for p in net.parameters():
        n += p.numel()

    return n

def main():
    global args
    # channels = 128, params = 17542026
    '''
    python online.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256

    python online.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -sop -resume /home/wfang/chinese_review/cf10/logs/pt/logs/online_T_4_e256_b128_sgd_lr0.1_c128_amp/checkpoint_max.pth
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
    parser.add_argument('-prefix', default='online', type=str)
    parser.add_argument('-BN-type', default='new', type=str)
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

    net = CIFAR10Net(args.channels, args.T)
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
        energy.get_sops_over_test_set_online(net, test_data_loader, args)

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
                # img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                loss = 0.
                net.set_init()
                y = None
                for t in range(args.T):
                    yt = net(img)
                    loss_t = F.cross_entropy(yt, label, label_smoothing=0.1) / args.T

                    if scaler is not None:
                        scaler.scale(loss_t).backward()
                        if t == args.T - 1:
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        if t == args.T - 1:
                            loss_t.backward()
                            optimizer.step()


                    loss = loss + loss_t.item()
                    y = yt if y is None else y + yt
                # y = net(img).mean(0)



            train_samples += label.shape[0]
            train_loss += loss * label.shape[0]
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
                
                loss = 0.
                net.set_init()
                y = None
                for t in range(args.T):
                    yt = net(img)
                    loss = loss + F.cross_entropy(yt, label, label_smoothing=0.1) / args.T
                    y = yt if y is None else y + yt

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