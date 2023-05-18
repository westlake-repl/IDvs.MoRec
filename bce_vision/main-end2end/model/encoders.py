import torch
import torch.nn as nn
from .modules import TransformerEncoder
from torch.nn.init import xavier_normal_, constant_


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
        return self.activate(self.image_net(item_content)[0])


class Resnet_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Resnet_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        return self.activate(self.image_net(item_content))


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
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)

