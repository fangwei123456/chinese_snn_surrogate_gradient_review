from datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
from config import Config
# from snn_delays import SnnDelays
import torch
from snn import SNN
import utils
import argparse
import os

'''
python main.py -sn spsn4 -norm bn -load /home/wfang/chinese_review/shd/logs/if_tebn/best_acc_113_0.7454025205761317.pt -sop
'''
parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
parser.add_argument('-device', default='cuda:0', help='device')
parser.add_argument('-sn', default='lif')
parser.add_argument('-norm', default='bn')
parser.add_argument('-out-dur', default='./logs')
parser.add_argument('-load', default='')
parser.add_argument('-sop', action='store_true')

args = parser.parse_args()
print(args)
out_dir = os.path.join(args.out_dur, f'{args.sn}_{args.norm}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
device = args.device
print(f"\n=====> Device = {device} \n\n")

config = Config()

if config.model_type == 'snn':
    model = SNN(config, args).to(device)
else:
    raise ValueError
    # model = SnnDelays(config).to(device)

if config.model_type == 'snn_delays_lr0':
    model.round_pos()

print(model)
'''
SNN(
  (model): Sequential(
    (0): Linear(in_features=140, out_features=256, bias=False)
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=m)
    (2): LIFNode(
      v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=m, backend=torch, tau=1.0050000001000001
      (surrogate_function): ATan(alpha=5.0, spiking=True)
    )
    (3): Dropout(p=0.4)
    (4): Linear(in_features=256, out_features=256, bias=False)
    (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=m)
    (6): LIFNode(
      v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=m, backend=torch, tau=1.0050000001000001
      (surrogate_function): ATan(alpha=5.0, spiking=True)
    )
    (7): Dropout(p=0.4)
    (8): Linear(in_features=256, out_features=20, bias=False)
    (9): LIFNode(
      v_threshold=1000000000.0, v_reset=0.0, detach_reset=True, step_mode=m, backend=torch, tau=1.0050000001000001
      (surrogate_function): ATan(alpha=5.0, spiking=True)
    )
  )
)
'''
print(f"===> Dataset    = {config.dataset}")
print(f"===> Model type = {config.model_type}")
print(f"===> Model size = {utils.count_parameters(model)}\n\n")


if args.sn == 'osr':
    import review_modules
    def osr_init_hook(m, input):
        for sm in m.modules():
            if isinstance(sm, (review_modules.OnlineLIFNode, review_modules.OSR)):
                sm.init = True


    model.register_forward_pre_hook(osr_init_hook)
    print('register_forward_pre_hook for online modules')



if config.dataset == 'shd':
    train_loader, valid_loader = SHD_dataloaders(config)
    test_loader = None
elif config.dataset == 'ssc':
    train_loader, valid_loader, test_loader = SSC_dataloaders(config)
elif config.dataset == 'gsc':
    train_loader, valid_loader, test_loader = GSC_dataloaders(config)
else:
    raise Exception(f'dataset {config.dataset} not implemented')

# if args.sn == 'blockalif':
#     # 生成神经元的数量
#     with torch.no_grad():
#         model.eval()
#         for x, y, _ in train_loader:
#             x = x.permute(1, 0, 2).float().to(device)  # (time, batch, neurons)
#             model(x)
#             model.reset_model(train=False)
#             model.train()
#             break


# T_min = 1e9
# T_max = 0
# for x, y, _ in train_loader:
#     T = x.shape[1]
#     T_min = min(T_min, T)
#     T_max = max(T_max, T)
#
# for x, y, _ in valid_loader:
#     T = x.shape[1]
#     T_min = min(T_min, T)
#     T_max = max(T_max, T)
#
# print(T_min, T_max)  # 88, 126

if args.sop:
    import numpy as np
    import torch
    import torch.nn as nn


    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.flops = m.in_features * m.out_features


    def record_spike_hook(module, input):
        with torch.no_grad():
            module.in_spikes += input[0].sum()
            module.in_spikes_numel += input[0].numel()


    def set_record_spike(net: nn.Module):
        hds = []
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.AvgPool1d, nn.AvgPool2d)):
                m.in_spikes = 0
                m.in_spikes_numel = 0
                hds.append(m.register_forward_pre_hook(record_spike_hook))
        return hds


    def get_sops(net: nn.Module):

        # 首先将in_spikes统一到[N, -1]的shape
        in_spikes_list = []
        in_spikes_numel_list = []
        flops_list = []

        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    in_spikes = m.in_spikes
                    in_spikes_numel = m.in_spikes_numel
                    in_spikes_list.append(in_spikes)
                    in_spikes_numel_list.append(in_spikes_numel)
                    flops_list.append(m.flops)

        sops = 0
        for i in range(len(in_spikes_list)):
            # s.shape = [N, -1]
            flops = flops_list[i]
            # print(in_spikes_list[i])
            fr = in_spikes_list[i] / in_spikes_numel_list[i]
            sops += (fr * flops).sum().item()

        return sops

    model.load_state_dict(torch.load(args.load, map_location='cpu'))
    set_record_spike(model)
    model.eval_model(valid_loader, device)
    print('sop=', get_sops(model) / 1e6)
    exit()

model.train_model(train_loader, valid_loader, test_loader, device, out_dir)