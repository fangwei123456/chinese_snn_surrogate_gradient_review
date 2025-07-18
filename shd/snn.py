import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional

from model import Model
from utils import set_seed
import review_modules
class SNN(Model):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.config = config
        self.args = args
    # Try factoring this method
    # Check ThresholdDependent batchnorm (in spikingjelly)
    def build_model(self):

        ########################### Model Description :
        #
        #  self.blocks = (n_layers,  0:weights+bn  |  1: lif+dropout+(synapseFilter) ,  element in sub-block)
        #


        ################################################   First Layer    #######################################################

        self.blocks = [[[layer.Linear(self.config.n_inputs, self.config.n_hidden_neurons, bias = self.config.bias, step_mode='m')],
                        [layer.Dropout(self.config.dropout_p, step_mode='m')]]]
        
        if self.config.use_batchnorm: self.blocks[0][0].insert(1, review_modules.create_norm(norm=self.args.norm, channels=self.config.n_hidden_neurons))

        if self.args.sn == 'lif':
            self.blocks[0][1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))

        else:
            self.blocks[0][1].insert(0, review_modules.create_neuron(neu=self.args.sn, tau=self.config.init_tau, surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset))


        if self.config.stateful_synapse:
            self.blocks[0][1].append(layer.SynapseFilter(tau=self.config.stateful_synapse_tau, learnable=self.config.stateful_synapse_learnable, 
                                                         step_mode='m'))


        ################################################   Hidden Layers    #######################################################

        for i in range(self.config.n_hidden_layers-1):
            self.block = [[layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias = self.config.bias, step_mode='m')],
                            [layer.Dropout(self.config.dropout_p, step_mode='m')]]
        
            if self.config.use_batchnorm: self.block[0].insert(1, review_modules.create_norm(norm=self.args.norm, channels=self.config.n_hidden_neurons))
            if self.args.sn == 'lif':
                self.block[1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            else:
                self.block[1].insert(0, review_modules.create_neuron(neu=self.args.sn, tau=self.config.init_tau, surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset))
            
            if self.config.stateful_synapse:
                self.block[1].append(layer.SynapseFilter(tau=self.config.stateful_synapse_tau, learnable=self.config.stateful_synapse_learnable, 
                                                             step_mode='m'))

            self.blocks.append(self.block)


        ################################################   Final Layer    #######################################################


        self.final_block = [[layer.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias = self.config.bias, step_mode='m')]]
        self.final_block.append([neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.output_v_threshold,
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True)])




        self.blocks.append(self.final_block)

        self.model = [l for block in self.blocks for sub_block in block for l in sub_block]
        self.model = nn.Sequential(*self.model)
        #print(self.model)

        self.positions = []
        self.weights = []
        self.weights_bn = []
        self.weights_plif = []
        for m in self.model.modules():
            if isinstance(m, layer.Linear):
                self.weights.append(m.weight)
                if self.config.bias:
                    self.weights_bn.append(m.bias)
            elif isinstance(m, layer.BatchNorm1d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)
            elif isinstance(m, neuron.ParametricLIFNode):
                self.weights_plif.append(m.w)





    def init_model(self):

        set_seed(self.config.seed)
        self.mask = []

        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers+1):
                # can you replace with self.weights ?
                torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].weight, nonlinearity='relu')
                if self.config.sparsity_p > 0:
                    with torch.no_grad():
                        self.mask.append(torch.rand(self.blocks[i][0][0].weight.size()).to(self.blocks[i][0][0].weight.device))
                        self.mask[i][self.mask[i]>self.config.sparsity_p]=1
                        self.mask[i][self.mask[i]<=self.config.sparsity_p]=0
                        #self.blocks[i][0][0].weight = torch.nn.Parameter(self.blocks[i][0][0].weight * self.mask[i])
                        self.blocks[i][0][0].weight *= self.mask[i]



    def reset_model(self, train=True):
        functional.reset_net(self)
        for i in range(self.config.n_hidden_layers+1):                
            if self.config.sparsity_p > 0:
                with torch.no_grad():
                    self.mask[i] = self.mask[i].to(self.blocks[i][0][0].weight.device)
                    #self.blocks[i][0][0].weight = torch.nn.Parameter(self.blocks[i][0][0].weight * self.mask[i])
                    self.blocks[i][0][0].weight *= self.mask[i]


    def decrease_sig(self, epoch):
        pass



    def forward(self, x):
        
        for block_id in range(self.config.n_hidden_layers):

            x = self.blocks[block_id][0][0](x)

            if self.config.use_batchnorm:
                # we use x.unsqueeze(3) to respect the expected shape to batchnorm which is (time, batch, channels, length)
                # we do batch norm on the channels since length is the time dimension
                # we use squeeze to get rid of the channels dimension 
                x = self.blocks[block_id][0][1](x.unsqueeze(3)).squeeze()
            
            # we use our spiking neuron filter
            spikes = self.blocks[block_id][1][0](x)
            # we use dropout on generated spikes tensor
            x = self.blocks[block_id][1][1](spikes)
            # we apply synapse filter
            if self.config.stateful_synapse:
                x = self.blocks[block_id][1][2](x)

        
        # Apply final layer
        out = self.blocks[-1][0][0](x)
        out = self.blocks[-1][1][0](out)

        if self.config.loss != 'spike_count':
            out = self.blocks[-1][1][0].v_seq

        return out#, self.blocks[0][1][0].v_seq
    


