# @Time: 2022.3.29 17:15
# @Author: Bolun Wu

import torch
from torch import nn

from configs import config


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0, std=config.astddev)
        if m.bias is not None: 
            m.bias.data.normal_(mean=0, std=config.bstddev)


class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.block = nn.ModuleList()
        for i in range(len(config.full_net) - 2):
            self.block.append(nn.Linear(config.full_net[i], config.full_net[i+1]))
            self.block.append(nn.Tanh())
        i = len(config.full_net) - 2
        self.block.append(nn.Linear(config.full_net[i], config.full_net[i+1], bias=False))
        
    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x

if __name__ == '__main__':
    model = NetWork()
    # export onnx format which can be visualized on Netron
    torch.onnx.export(model, torch.rand([8, 1]), 'model.onnx', input_names=['input'], output_names=['output'])
