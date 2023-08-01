import torch
import torch.nn as nn
from .modules import TransformerEncoder
from torch.nn.init import xavier_normal_, constant_


class MLP_Layers(torch.nn.Module):
    def __init__(self, word_embedding_dim, item_embedding_dim, layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=drop_rate))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.GELU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        sample_items = self.activate(self.fc(sample_items))
        return self.mlp_layers(sample_items)


class Swin_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Swin_Encoder, self).__init__()
        self.image_net = image_net

    def forward(self, item_content):
        return self.image_net(item_content)[1]


class Resnet_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Resnet_Encoder, self).__init__()
        self.image_net = image_net

    def forward(self, item_content):
        return self.image_net(item_content)


class User_Encoder(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(User_Encoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1)))
        # att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)

