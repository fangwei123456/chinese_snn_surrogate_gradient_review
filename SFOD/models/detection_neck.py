from torch import nn
import torch

from models.utils import SpikingNeckBlock
from models.SSD_utils import init_weights


class DetectionNeck(nn.Module):
    def __init__(self, in_channels, fusion_layers=4, T:int=5, norm:str='bn', sn:str='lif'):
        super().__init__()

        self.nz, self.numel = {}, {}

        # Define the upsampling module
        self.ft_module = nn.ModuleList(
            [
                SpikingNeckBlock(in_channels[0], 128, up_flag=False, T=T, norm=norm, sn=sn),
                SpikingNeckBlock(in_channels[1], 128, kernel_size=4, stride=2, padding=1, T=T, norm=norm, sn=sn),
                SpikingNeckBlock(in_channels[2], 128, kernel_size=8, stride=4, padding=1, T=T, norm=norm, sn=sn),
            ]
        )
        if fusion_layers == 4:
            self.ft_module.append(SpikingNeckBlock(in_channels[3], 128, kernel_size=(11, 12), stride=7, padding=1, T=T, norm=norm, sn=sn))

        self.ft_module.apply(init_weights)
        self.out_channel = 128 * fusion_layers

    def forward(self, source_features):
        assert len(source_features) == len(self.ft_module)
        transformed_features = []

        # upsample
        for k, v in enumerate(self.ft_module):
            transformed_features.append(v(source_features[k].permute(1, 0, 2, 3, 4)))  # Before inputting, change to T N C H W

        concat_fea = torch.cat(transformed_features, 2)

        return concat_fea.permute(1, 0, 2, 3, 4)  # N T C H W

    