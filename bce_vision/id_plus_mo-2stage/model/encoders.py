import torch
import torch.nn as nn
from .modules import TransformerEncoder
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

class ADD(torch.nn.Module):
    def __init__(self, ):
        super(ADD, self).__init__()

    def forward(self, x, y):
        return x + y







class CAT(torch.nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(CAT, self).__init__()
        mlp_modules = []
        mlp_modules.append(nn.Dropout(p=drop_rate))
        mlp_modules.append(nn.Linear(input_dim, output_dim))
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

    def forward(self, x, y):
        con_cat = torch.cat([x, y], 1)
        return self.mlp_layers(con_cat)



class FC_Layers(torch.nn.Module):
    def __init__(self, word_embedding_dim, item_embedding_dim, dnn_layers, drop_rate):
        super(FC_Layers, self).__init__()
        self.dnn_layers = dnn_layers
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()

        if self.dnn_layers > 0:
            self.mlp_layers = MLP_Layers(layers=[item_embedding_dim] * (self.dnn_layers + 1),
                                  dnn_layers=self.dnn_layers,
                                  drop_rate=drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        x = self.activate(self.fc(sample_items))
        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        else:
            return x

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

