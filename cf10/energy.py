import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.activation_based import functional, layer

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
                conv.append(nn.Conv1d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(nn.BatchNorm1d(channels))
                conv.append(nn.ReLU())

            conv.append(nn.AvgPool1d(2, 2))

        self.conv = nn.Sequential(*conv)


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8, channels * 8 // 4),
            nn.ReLU(),
            nn.Linear(channels * 8 // 4, 10),
        )

    def forward(self, x_seq: torch.Tensor):
        x_seq = x_seq.permute(3, 0, 1, 2)
        x_seq = x_seq.flatten(0, 1)
        return self.fc(self.conv(x_seq))

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



def record_input_shape_hook(module, input, output):
    if hasattr(module, 'step_mode') and module.step_mode == 'm':
        # 忽略时间维度
        module.input_shape = input[0].shape[1:]
        module.output_shape = output[0].shape[1:]
    else:
        module.input_shape = input[0].shape
        module.output_shape = output[0].shape





def set_input_shape(net: nn.Module, x: torch.Tensor):
    '''

    给模块设置钩子，然后给与输入，记录下突触层的输入shape，便于之后计算flop

    '''
    hds = []
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.AvgPool2d, nn.AvgPool1d)):
            hds.append(m.register_forward_hook(record_input_shape_hook))

    with torch.no_grad():
        net(x)
        functional.reset_net(net)

    for h in hds:
        h.remove()




def set_flops(net: CIFAR10Net):
    for i in range(len(net.conv)):
        m = net.conv[i]

        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            m = net.conv[i]
            if isinstance(net.conv[i - 1], (nn.AvgPool1d, nn.AvgPool2d)):
                # 假定最大池化和卷积已经合并了
                m.flops = conv_syops_counter(m, net.conv[i - 1].input_shape, m.output_shape)
            else:
                m.flops = conv_syops_counter(m, m.input_shape, m.output_shape)
            print(m, 'flops=', m.flops)


    for m in net.fc.modules():
        if isinstance(m, nn.Linear):
            m.flops = linear_syops_counter(m, m.input_shape, m.output_shape)
            print(m, 'flops=', m.flops)




def record_spike_hook(module, input):
    with torch.no_grad():
        module.in_spikes += input[0].sum()
        module.in_spikes_numel += input[0].numel()


def set_record_spike(net: nn.Module, ignore_layers: tuple):
    hds = []
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.AvgPool1d, nn.AvgPool2d)):
            if m not in ignore_layers:
                m.in_spikes = 0
                m.in_spikes_numel = 0
                hds.append(m.register_forward_pre_hook(record_spike_hook))
    return hds


def get_sops(net: CIFAR10Net):

    # 首先将in_spikes统一到[N, -1]的shape
    in_spikes_list = []
    in_spikes_numel_list = []
    flops_list = []
    with torch.no_grad():
        for i in range(1, len(net.conv)):
            # conv[0]是直接输入图片
            m = net.conv[i]
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
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

        for m in net.fc.modules():
            if isinstance(m, nn.Linear):
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
            img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

            set_input_shape(net, img[:, 0:1])  # 获取每一层输入输出的shape
            set_flops(net)  # 根据shape计算flop
            functional.reset_net(net)
            break

        set_record_spike(net, ignore_layers=(net.conv[0], ))
        numel = 0
        test_acc = 0.
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
            y = net(img).mean(0)
            numel += label.numel()
            test_acc += (y.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
        print(test_acc / numel)
        print(f'flops={net.conv[0].flops / 1e6:.3f}, sops={get_sops(net) * args.T / 1e6:.3f}')




def get_sops_over_test_set_online(net: nn.Module, test_data_loader, args):
    net.eval()
    with torch.no_grad():

        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            net.set_init()
            set_input_shape(net, img[0:1])  # 获取每一层输入输出的shape
            set_flops(net)  # 根据shape计算flop
            functional.reset_net(net)
            break

        set_record_spike(net, ignore_layers=(net.conv[0], ))
        numel = 0
        test_acc = 0.
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            net.set_init()
            y = None
            for t in range(args.T):
                yt = net(img)
                y = yt if y is None else y + yt


            numel += label.numel()
            test_acc += (y.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
        print(test_acc / numel)
        print(f'flops={net.conv[0].flops / 1e6:.3f}, sops={get_sops(net) * args.T / 1e6:.3f}')

if __name__ == '__main__':

    net = CIFAR10Net(16)

    set_input_shape(net, torch.randn(16, 3, 32, 32))

    set_flops(net)