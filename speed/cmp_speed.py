import torch
from spikingjelly.activation_based.cuda_utils import cal_fun_t
from spikingjelly.activation_based import surrogate, neuron

device = 'cuda:1'
repeats = 64

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



def create_block_alif(T:int, block_size: int):
    #TODO 加上BlockALIF神经元的代码
    # T: 仿真步数
    # block_size: 并行block的大小
    # 该函数返回一个block alif神经元
    pass

# powers = (8, 12, 16, 20)
# Ts = (2, 4, 8, 16, 32, 64)
powers = (20, )
Ts = (20, )
lif_cp.train()
print('train')
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

        print(f'{power}, {T}, {t_lif}, {t_psn}, {t_lif_cp}')

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
            print(f'{power}, {T}, {t_lif}, {t_psn}, {t_lif_cp}')




