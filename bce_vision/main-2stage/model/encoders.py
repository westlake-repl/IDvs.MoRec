import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_


class MLP_Layers(torch.nn.Module):
    def __init__(self, layers, dnn_layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        self.dnn_layers = dnn_layers
        if self.dnn_layers > 0:
            mlp_modules = []
            for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x):
        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        else:
            return x


class FC_Layers(torch.nn.Module):
    def __init__(self, word_embedding_dim, item_embedding_dim):
        super(FC_Layers, self).__init__()
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            constant_(self.fc.bias.data, 0)
        self.activate = nn.GELU()

    def forward(self, sample_items):
        return self.activate(self.fc(sample_items))


class MAE_Encoder(torch.nn.Module):
    def __init__(self, image_net, item_dim):
        super(MAE_Encoder, self).__init__()
        self.item_dim = item_dim
        self.word_emb = 768
        self.image_net = image_net
        self.activate = nn.GELU()

        self.cv_proj = nn.Linear(self.word_emb, self.item_dim)
        xavier_normal_(self.cv_proj.weight.data)
        if self.cv_proj.bias is not None:
            constant_(self.cv_proj.bias.data, 0)

    def forward(self, item_content):
        return self.activate(self.cv_proj(self.image_net(item_content)[0][:, 0]))


class Vit_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Vit_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        return self.activate(self.image_net(item_content)[1])


class Resnet_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Resnet_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        return self.activate(self.image_net(item_content))
