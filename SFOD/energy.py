import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.activation_based import functional, layer


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
    module.input_shape = input[0].shape
    module.output_shape = output.shape
    # print(module, module.input_shape, module.output_shape)





def set_input_shape(net: nn.Module, x: torch.Tensor):
    '''

    给模块设置钩子，然后给与输入，记录下突触层的输入shape，便于之后计算flop

    '''
    hds = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            hds.append(m.register_forward_hook(record_input_shape_hook))

    with torch.no_grad():
        T, N, C, H, W = x.shape
        assert T == 1 and N == 1
        net(x)
        functional.reset_net(net)

    for h in hds:
        h.remove()




def set_flops(net: nn.Module):
    for i, m in enumerate(net.modules()):

        if isinstance(m, nn.Conv2d) and hasattr(m, 'input_shape'):

            m.flops = conv_syops_counter(m, m.input_shape, m.output_shape)
            # print(m, 'flops=', m.flops)





def record_spike_hook(module, input):
    with torch.no_grad():
        module.in_spikes += input[0].sum()
        module.in_spikes_numel += input[0].numel()

from models import review_modules
def set_record_spike(net: nn.Module):
    hds = []
    mds = list(net.modules())
    for i in range(len(mds)):
        if isinstance(mds[i], nn.Conv2d):
            # 网络结构中是bn/conv，因此bn的输入才是脉冲
            m = None
            if isinstance(mds[i - 1], (nn.BatchNorm2d, review_modules.OSR)):
                m = mds[i - 1]
            elif isinstance(mds[i - 1], (layer.SeqToANNContainer, layer.MultiStepContainer)):
                if isinstance(mds[i - 2], (nn.BatchNorm2d, review_modules.OSR)):
                    m = mds[i - 2]
            if m is None:
                raise ValueError

            m.in_spikes = 0
            m.in_spikes_numel = 0
            hds.append(m.register_forward_pre_hook(record_spike_hook))

    return hds


def get_sops(net: nn.Module):

    # 首先将in_spikes统一到[N, -1]的shape
    in_spikes_list = []
    in_spikes_numel_list = []
    flops_list = []
    mds = list(net.modules())
    with torch.no_grad():
        for i in range(len(mds)):

            if isinstance(mds[i], nn.Conv2d) and hasattr(mds[i], 'flops'):
                m = None
                if isinstance(mds[i - 1], (nn.BatchNorm2d, review_modules.OSR)):
                    m = mds[i - 1]
                elif isinstance(mds[i - 1], (layer.SeqToANNContainer, layer.MultiStepContainer)):
                    if isinstance(mds[i - 2], (nn.BatchNorm2d, review_modules.OSR)):
                        m = mds[i - 2]
                if m is None:
                    raise ValueError


                in_spikes = m.in_spikes
                in_spikes_numel = m.in_spikes_numel
                in_spikes_list.append(in_spikes)
                in_spikes_numel_list.append(in_spikes_numel)
                # 网络结构中是bn/conv，因此bn的输入才是脉冲

                flops_list.append(mds[i].flops)


    sops = 0
    for i in range(len(in_spikes_list)):
        # s.shape = [N, -1]
        flops = flops_list[i]
        # print(in_spikes_list[i])
        fr = in_spikes_list[i] / in_spikes_numel_list[i]
        sops += (fr * flops).sum().item()

    return sops



