import torch
from torch import nn
from .encoders import Bert_Encoder, FC_Layers, MLP_Layers
from torch.nn.init import xavier_normal_


class Model(torch.nn.Module):

    def __init__(self, args, user_num, item_num, use_modal):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.embedding_dim = args.embedding_dim
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.neg_num = 1

        self.user_embedding = nn.Embedding(user_num + 1, self.embedding_dim, padding_idx=0)
        xavier_normal_(self.user_embedding.weight.data)

        if self.use_modal:
            self.turn_dim = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                      item_embedding_dim=self.embedding_dim)
            self.fc = MLP_Layers(layers=[self.embedding_dim] * (self.dnn_layers+1),
                                 dnn_layers=self.dnn_layers,
                                 drop_rate=args.drop_rate)
        else:
            self.id_embedding = nn.Embedding(item_num + 1, self.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_embedding.weight.data)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_user, input_items, bce_label):
        user_embedding = self.user_embedding(input_user)
        if self.use_modal:
            input_embs = self.fc(self.turn_dim(input_items))
        else:
            input_embs = self.id_embedding(input_items)
        item_feature = input_embs.view(-1, 1 + self.neg_num, self.embedding_dim)
        score = torch.bmm(item_feature, user_embedding.unsqueeze(-1)).squeeze(dim=-1)
        loss = self.criterion(score.view(-1), bce_label.view(-1))
        return loss
