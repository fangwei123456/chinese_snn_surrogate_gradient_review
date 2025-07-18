import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import layer, neuron, surrogate, base, functional


class TEBN(nn.Module):
    def __init__(self, T, num_features):
        super(TEBN, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.p = nn.Parameter(torch.ones(T, 1, 1, 1))
    def forward(self, x):
        y = functional.seq_to_ann_forward(x, self.bn) * self.p
        return y

class SlidingPSN(base.MemoryModule):

    @property
    def supported_backends(self):
        return 'gemm', 'conv'

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.k)
            length = min(end - start, self.k)
            weight[i][start: end] = self.weight[self.k - length: self.k]

        return weight

    def __init__(self, k: int, exp_init: bool = True,
                 surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(), step_mode: str = 's',
                 backend: str = 'gemm'):


        super().__init__()
        self.register_memory('queue', [])
        self.step_mode = step_mode
        self.k = k
        self.surrogate_function = surrogate_function
        self.backend = backend

        if exp_init:
            weight = torch.ones([k])
            for i in range(k - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.
        else:
            weight = torch.ones([1, k])
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight = weight[0]

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.as_tensor(-1.))

    def single_step_forward(self, x: torch.Tensor):
        self.queue.append(x.flatten())
        if self.queue.__len__() > self.k:
            self.queue.pop(0)

        weight = self.weight[self.k - self.queue.__len__(): self.k]
        x_seq = torch.stack(self.queue)

        for i in range(x.dim()):
            weight = weight.unsqueeze(-1)

        h = torch.sum(weight * x_seq, 0)
        spike = self.surrogate_function(h + self.bias)

        return spike.view(x.shape)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'gemm':

            weight = self.gen_gemm_weight(x_seq.shape[0])
            h_seq = torch.addmm(self.bias, weight, x_seq.flatten(1)).view(x_seq.shape)
            return self.surrogate_function(h_seq)
        elif self.backend == 'conv':

            # x_seq.shape = [T, N, *]
            x_seq_shape = x_seq.shape
            # [T, N, *] -> [T, N] -> [N, T] -> [N, 1, T]
            x_seq = x_seq.flatten(1).t().unsqueeze(1)

            x_seq = F.pad(x_seq, pad=(self.k - 1, 0))
            x_seq = F.conv1d(x_seq, self.weight.view(1, 1, -1), stride=1)

            x_seq = x_seq.squeeze(1).t().view(x_seq_shape)
            return self.surrogate_function(x_seq + self.bias)

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', order={self.k}'
class CLIFSpike(nn.Module):
    def __init__(self, tau, surrogate_function):
        super(CLIFSpike, self).__init__()
        # the symbol is corresponding to the paper
        # self.spike_func = surrogate_function
        self.spike_func = surrogate_function

        self.v_th = 1.
        self.gamma = 1 - 1. / tau

    def forward(self, x_seq):
        # x_seq.shape should be [T, N, *]
        _spike = []
        u = 0
        m = 0
        T = x_seq.shape[0]
        for t in range(T):
            u = self.gamma * u + x_seq[t, ...]
            spike = self.spike_func(u - self.v_th)
            _spike.append(spike)
            m = m * torch.sigmoid_((1. - self.gamma) * u) + spike
            u = u - spike * (self.v_th + torch.sigmoid_(m))
        # self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(_spike, dim=0)

class PSN(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
        super().__init__()
        self.T = T
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)

    def extra_repr(self):
        return super().extra_repr() + f'T={self.T}, '

class TandemIFATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, v=0., v_th=1.):

        s_seq = []
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            s = (v >= v_th).to(x_seq)
            v = v - s * v_th
            s_seq.append(s)

        s_seq = torch.stack(s_seq, dim=0)
        return s_seq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class TandemIF(nn.Module):
    def __init__(self):
        super(TandemIF, self).__init__()
    def forward(self, x_seq):
        return TandemIFATGF.apply(x_seq)


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
    def __init__(self, n_in, n_out, t_len, t_latency, recurrent=False, beta_grad=True, adapt=True, init_beta=1,
                 init_p=1, detach_spike_grad=True, surr_grad=FastSigmoid.apply):
        super(Blocks, self).__init__()
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
        self._p_ident_base = nn.Parameter(torch.ones(self._t_len_block, n_out), requires_grad=False)
        self._p_exp = nn.Parameter(torch.arange(1, self._t_len_block + 1).float().unsqueeze(1), requires_grad=False)
        if recurrent == True:
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

    def forward(self, x):  # 输入[t,b,n]
        if self._t_pad != 0:
            x = F.pad(x, pad=(0, 0, 0, 0, 0, self._t_pad))
        a_kernel = torch.zeros_like(x).to(x.device)[:self._t_len_block, :, :]
        v_th = torch.ones_like(x).to(x.device)[:self._t_len_block, :, :]
        spikes_list = []
        v_init = torch.zeros_like(x[0, :, :]).to(x.device)
        int_mem = torch.zeros_like(x[0, :, :]).to(x.device)
        for i in range(self._n_blocks):
            # x_slice = x[:, :,i * self._t_len_block: (i+1) * self._t_len_block].contiguous()
            # print(i * self._t_len_block,(i+1) * self._t_len_block)
            x_slice = x[i * self._t_len_block: (i + 1) * self._t_len_block, :, :]
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
                    v_th = 1 + self.b.view(1, 1, -1) * a_kernel  # 较耗时间
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
        self._beta_exp = nn.Parameter(torch.arange(t_len).flip(0).unsqueeze(0).expand(n_in, t_len).float(),
                                      requires_grad=False)
        self._phi_kernel = nn.Parameter((torch.arange(t_len) + 1).flip(0).float().view(1, 1, 1, t_len),
                                        requires_grad=False)

    @staticmethod
    def g(faulty_spikes):
        negate_faulty_spikes = faulty_spikes.clone().detach()
        negate_faulty_spikes[faulty_spikes == 1.0] = 0
        faulty_spikes -= negate_faulty_spikes
        return faulty_spikes

    def forward(self, current, beta, v_init=None, v_th=1):
        current = current.permute(1, 2, 0).contiguous()
        v_th = v_th.permute(1, 2, 0).contiguous()

        if v_init is not None:
            current[:, :, 0] += beta * v_init  # 较耗时间
        pad_current = F.pad(current, pad=(self._t_len - 1, 0))
        beta_kernel = self.build_beta_kernel(beta)  # 不耗时间
        b, n, t = pad_current.shape
        n, in_channels, kernel_width_size = beta_kernel.shape
        membrane = F.conv1d(pad_current, weight=beta_kernel, bias=None, stride=1, dilation=1, groups=n,
                            padding=0)  # 很耗时间
        faulty_spikes = self._surr_grad(membrane - v_th)
        pad_spikes = F.pad(faulty_spikes, pad=(self._t_len - 1, 0)).unsqueeze(1)
        z = F.conv2d(pad_spikes, self._phi_kernel)  # 很耗时间
        z = z.squeeze(1).permute(2, 0, 1).contiguous()
        z_copy = z.clone()
        return Block.g(z), z_copy, membrane[:, :, -1].contiguous()

    def build_beta_kernel(self, beta):
        return torch.pow(beta.unsqueeze(1), self._beta_exp).unsqueeze(1)


class ALIF(nn.Module):
    def __init__(self, n_in, n_out, t_len, t_latency, recurrent=False, beta_grad=True, adapt=True, init_beta=1,
                 init_p=1, detach_spike_grad=True, surr_grad=FastSigmoid.apply):
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

    def get_rec_input(self, spikes):  # 计算递归输入，结合当前脉冲活动和递归权重
        return torch.einsum("ij, bj...->bi...", self.rec_weight, spikes.detach() if self._detach_spike_grad else spikes)

    @property
    def beta(self):  # 保证beta在合理范围内
        return torch.clamp(self._beta, min=0.001, max=0.999)

    @property
    def p(self):  # 保证自适应系数p在合理范围内
        return torch.clamp(self._p.abs(), min=0, max=0.999)

    @property
    def b(self):  # 保证自适应系数b在合理范围内
        return torch.clamp(self._b.abs(), min=0.001, max=1)

    def forward(self, x):  # x: 输入张量，形状为 (t, b, n) 表示时间步、批量大小、神经元数
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


class OSR(nn.Module):
    def __init__(self, T, num_features):
        super(OSR, self).__init__()
        self.eps = 1e-5
        self.init = False
        self.shape = [1, num_features, 1]

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('run_mean', torch.zeros(num_features))
        self.register_buffer('run_var', torch.ones(num_features))
        # for estimating total mean and var
        self.total_mean = 0.
        self.total_var = 0.
        self.momentum = 1 - (1 - 0.9) / T
        self.T = T
        self.BN_type = 'new'


    def forward(self, x):
        eps = self.eps

        if self.training and self.init and isinstance(self.total_var, torch.Tensor):
            with torch.no_grad():
                mean = self.total_mean / self.T
                var = self.total_var / self.T
                if self.BN_type == 'new': var -= mean ** 2
                self.run_mean += (1 - self.momentum) * (mean - self.run_mean)
                self.run_var += (1 - self.momentum) * (var - self.run_var)
                self.total_mean = 0.
                self.total_var = 0.

        # count_all = torch.full((1,), x.numel() // x.size(1), dtype = x.dtype, device=x.device)
        if self.training:
            dims = [0, 2]
            mean, var = x.mean(dim=dims), x.var(dim=dims)
            invstd = 1. / torch.sqrt(var + eps)

            with torch.no_grad():
                self.total_mean += mean
                self.total_var += var
                if self.BN_type == 'new': self.total_var += mean ** 2

            y1 = (x - mean.reshape(self.shape)) * invstd.reshape(self.shape)
            with torch.no_grad():
                scale = torch.sqrt(var + eps) / torch.sqrt(self.run_var + eps)
                if self.training and self.BN_type == 'new':
                    bound = 5.
                    scale = torch.clip(scale, 1. / bound, bound)
                shift = (mean - self.run_mean) * scale / torch.sqrt(var + eps)
            y = (y1 * scale.reshape(self.shape) + shift.reshape(self.shape)) * self.gamma.reshape(
                self.shape) + self.beta.reshape(self.shape)
        else:
            mean, invstd = self.run_mean, 1. / torch.sqrt(self.run_var + eps)
            y = torch.batch_norm_elemt(x, self.gamma, self.beta, mean, invstd, eps)

        self.init = False
        return y


class OnlineLIFNode(neuron.LIFNode):
    def __init__(self, T, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function = surrogate.Sigmoid(),
                 detach_reset: bool = True, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.dynamic_threshold = True
        self.fixed_test_threshold = True
        if self.dynamic_threshold and self.fixed_test_threshold:
            self.register_buffer('run_th', torch.ones(1))
            self.th_momentum = 1 - (1 - 0.9) / T
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


'''

osr是单步前反向传播，其网络结构需要单独实现
'''

class AutoPadding(nn.Module):
    def __init__(self, T, m):
        super(AutoPadding, self).__init__()
        self.T = T
        self.m = m

    def forward(self, x):
        # x.shape = [T, N, C]
        T = x.shape[0]
        assert T <= self.T
        if T < self.T:
            # print('before', x.shape)
            padding = torch.zeros([self.T - T] + list(x.shape[1:]), device=x.device)
            x = torch.cat((x, padding), dim=0)
            # print('after ', x.shape)

        return self.m(x)[0: T]

def create_neuron(neu: str, tau, surrogate_function, detach_reset):
    T = 126
    if neu == 'if':
        return neuron.IFNode(surrogate_function=surrogate_function, detach_reset=detach_reset, step_mode='m', backend='cupy')
    elif neu == 'lif':
        raise ValueError
    elif neu == 'tandem':
        return TandemIF()
    elif neu == 'relu':
        return nn.ReLU()
    elif neu == 'clif':
        return CLIFSpike(tau=tau, surrogate_function=surrogate_function)
    elif neu == 'psn':
        raise ValueError
    elif neu == 'spsn2':
        return SlidingPSN(k=2, surrogate_function=surrogate_function, step_mode='m')
    elif neu == 'spsn3':
        return SlidingPSN(k=3, surrogate_function=surrogate_function, step_mode='m')
    elif neu == 'spsn4':
        return SlidingPSN(k=4, surrogate_function=surrogate_function, step_mode='m')
    elif neu == 'blockalif':
        return AutoPadding(T, DynamicReshapeModule(Blocks(n_in=256, n_out=256, t_len=T, t_latency=1, recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99, surr_grad=surrogate_function, detach_spike_grad=detach_reset)))
        # return BALIFWrapper(t_len=T, t_latency=1, surrogate_function=surrogate_function, detach_reset=detach_reset)
        # return DynamicReshapeModule(Blocks(n_in=kwargs['numel'], n_out=kwargs['numel'], t_len=T, t_latency=kwargs['lattency'], recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99))
    elif neu == 'osr':
        return layer.MultiStepContainer(OnlineLIFNode(T, tau=tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset))
    else:
        raise NotImplementedError


def create_norm(norm: str, channels):
    T = 126
    if norm == 'bn':
        return layer.BatchNorm1d(channels, step_mode='m')
    elif norm == 'tebn':
        return AutoPadding(T, TEBN(T, channels))
    elif norm == 'osr':
        return layer.MultiStepContainer(OSR(T, channels))
    else:
        raise ValueError

# DynamicReshapeModule(Blocks(n_in=channels*32*32, n_out=channels*32*32, t_len=T, t_latency=latency, recurrent=False,beta_grad=True,adapt=True,init_beta=0.99,init_p=0.99))