import torch
import torch.nn as nn
from spikingjelly.activation_based.cuda_utils import cal_fun_t
from spikingjelly.activation_based import surrogate, neuron
import math
import torch.nn.functional as F

device = 'cuda:0'
repeats = 64
# powers = (8, 12, 16, 20)
# Ts = (2, 4, 8, 16, 32, 64)
t_latencys = [0, 1, 3, 7]

powers = (8, 12, 16, )
Ts = (2, 4, 8, 16, 32, 64, )

@torch.jit.script
def lif_forward(x_seq: torch.Tensor, decay: float, v_reset: float = 0.):
    # x_seq.shape = [T, N]
    h_t = torch.zeros_like(x_seq[0])
    s_seq = []
    for t in range(x_seq.shape[0]):
        h_t = decay * x_seq[t] + (1. - decay) * h_t
        s_t = (h_t >= 1.).to(x_seq)
        s_seq.append(s_t)
        h_t = (1. - s_t) * h_t + v_reset * s_t
    return torch.stack(s_seq)

@torch.jit.script
def lif_charge(x: torch.Tensor, h: torch.Tensor, decay: float):
    return decay * x + (1. - decay) * h

@torch.jit.script
def lif_reset(s: torch.Tensor, h: torch.Tensor, v_reset: float):
    return (1. - s) * h + v_reset * s

sg = surrogate.ATan()

def lif_forward_backward(x_seq: torch.Tensor, decay: float, v_reset: float = 0.):
    # x_seq.shape = [T, N]
    h_t = torch.zeros_like(x_seq[0])
    s_seq = []
    for t in range(x_seq.shape[0]):
        h_t = lif_charge(x_seq[t], h_t, decay)
        s_t = sg(h_t - 1.)
        s_seq.append(s_t)
        h_t = lif_reset(s_t, h_t, v_reset)
    s_seq = torch.stack(s_seq)
    s_seq.sum().backward()
    x_seq.grad.zero_()


@torch.jit.script
def psn_forward(x_seq: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    # x_seq.shape = [N, T]
    # w.shape = [T, T]
    # b.shape = [T]
    h_seq = torch.addmm(b, x_seq, w)
    return (h_seq >= 0.).to(x_seq)


def psn_forward_backward(x_seq: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    # x_seq.shape = [N, T]
    # w.shape = [T, T]
    # b.shape = [T]
    h_seq = torch.addmm(b, x_seq, w)
    s_seq = sg(h_seq)
    s_seq.sum().backward()
    x_seq.grad.zero_()
    w.grad.zero_()
    b.grad.zero_()


lif_cp = neuron.LIFNode(surrogate_function=sg, step_mode='m', backend='cupy')

def lif_cp_forward(x_seq):
    s_seq = lif_cp(x_seq)
    lif_cp.reset()

def lif_cp_forward_backward(x_seq):
    s_seq = lif_cp(x_seq)
    s_seq.sum().backward()
    x_seq.grad.zero_()
    lif_cp.reset()


from blockalif_module import Blocks as BlockAlif

def blockalif_forward(x_seq, blockalif):

    return blockalif(x_seq)

def blockalif_forward_backward(x_seq, blockalif):

    s_seq = blockalif(x_seq)
    s_seq.sum().backward()
    x_seq.grad.zero_()

from blockalif_module import SNN as Alif
def alif_forward(x_seq, alif):

    return alif(x_seq)

def alif_forward_backward(x_seq, alif):

    s_seq = alif(x_seq)
    s_seq.sum().backward()
    x_seq.grad.zero_()





lif_cp.train()
print('train')

print('power, T, t_lif, t_psn, t_lif_cp, t_blockalif_0, t_blockalif_1, t_blockalif_2, t_blockalif_3, t_alif_0, t_alif_1, t_alif_2, t_alif_3')
for power in powers:
    N = 2 << power
    for T in Ts:
        x_seq = torch.rand([T, N], device=device, requires_grad=True)
        decay = 0.5
        t_lif = cal_fun_t(repeats, device, lif_forward_backward, x_seq, decay)

        x_seq = torch.rand([N, T], device=device, requires_grad=True)
        w = torch.rand([T, T], device=device, requires_grad=True)
        b = torch.rand([T], device=device, requires_grad=True)
        t_psn = cal_fun_t(repeats, device, psn_forward_backward, x_seq, w, b)

        x_seq = torch.rand([T, N], device=device, requires_grad=True)
        t_lif_cp = cal_fun_t(repeats, device, lif_cp_forward_backward, x_seq)

        x_seq = torch.rand([1, N, T], device=device, requires_grad=True)

        t_alif = []
        t_blockalif = []

        for t_latency in t_latencys:
            if t_latency + 1 >= T:
                t_alif.append(None)
                t_blockalif.append(None)
                continue

            blockalif = BlockAlif(n_in=N, n_out=N, rf_len=0, t_len=T,
                                  t_latency=t_latency, recurrent=False, init_beta=0.99, init_p=0.99).to(device)

            alif = Alif(n_in=N, n_out=N, rf_len=0, t_len=x_seq.shape[0], t_latency=t_latency,
                        recurrent=False, init_beta=0.99, init_p=0.99).to(device)


            t_alif.append(cal_fun_t(repeats, device, alif_forward_backward, x_seq, alif))
            t_blockalif.append(cal_fun_t(repeats, device, blockalif_forward_backward, x_seq, blockalif))

        print(f'{power}, {T}, {t_lif}, {t_psn}, {t_lif_cp}', end=',')
        for i in range(len(t_blockalif)):
            print(t_blockalif[i], end=',')

        for i in range(len(t_alif)):
            print(t_alif[i], end=',')
        print('')


lif_cp.eval()
print('inference')
with torch.no_grad():
    for power in powers:
        N = 2 << power
        for T in Ts:
            x_seq = torch.rand([T, N], device=device)
            decay = 0.5
            t_lif = cal_fun_t(repeats, device, lif_forward, x_seq, decay)

            x_seq = torch.rand([N, T], device=device)
            w = torch.rand([T, T], device=device)
            b = torch.rand([T], device=device)
            t_psn = cal_fun_t(repeats, device, psn_forward, x_seq, w, b)

            x_seq = torch.rand([T, N], device=device)
            t_lif_cp = cal_fun_t(repeats, device, lif_cp_forward, x_seq)

            x_seq = torch.rand([1, N, T], device=device)

            t_alif = []
            t_blockalif = []

            for t_latency in t_latencys:
                if t_latency + 1 >= T:
                    t_alif.append(None)
                    t_blockalif.append(None)
                    continue
                blockalif = BlockAlif(n_in=N, n_out=N, rf_len=0, t_len=T,
                                      t_latency=t_latency, recurrent=False, init_beta=0.99, init_p=0.99).to(device)

                alif = Alif(n_in=N, n_out=N, rf_len=0, t_len=T, t_latency=t_latency,
                            recurrent=False, init_beta=0.99, init_p=0.99).to(device)


                t_alif.append(cal_fun_t(repeats, device, alif_forward,x_seq, alif))
                t_blockalif.append(cal_fun_t(repeats, device, blockalif_forward,x_seq, blockalif))

            print(f'{power}, {T}, {t_lif}, {t_psn}, {t_lif_cp}', end=',')
            for i in range(len(t_blockalif)):
                print(t_blockalif[i], end=',')

            for i in range(len(t_alif)):
                print(t_alif[i], end=',')
            print('')
