import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_, constant_


class Vit_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Vit_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        return self.activate(self.image_net(item_content)[0])


class Resnet_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Resnet_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        return self.activate(self.image_net(item_content))


class MLP_Layers(torch.nn.Module):
    def __init__(self, layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=drop_rate))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.GELU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

    def forward(self, x):
        return self.mlp_layers(x)


class MLP_Encoder(torch.nn.Module):
    def __init__(self, embedding_dim, dnn_layers, drop_rate):
        super(MLP_Encoder, self).__init__()
        self.dnn_layers = dnn_layers
        if self.dnn_layers > 0:
            self.MLP = MLP_Layers(layers=[embedding_dim] * (self.dnn_layers + 1), drop_rate=drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, inputs):
        if self.dnn_layers > 0:
            return self.MLP(inputs)
        else:
            return inputs
