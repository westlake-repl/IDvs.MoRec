import torch
from torch import nn
from .encoders import Bert_Encoder, FC_Layers, User_Encoder, ADD, CAT, MLP_Layers
from torch.nn.init import xavier_normal_


class Model(torch.nn.Module):

    def __init__(self, args, item_num, use_modal, bert_model):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers

        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        self.turn_dim = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim*2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                              dnn_layers=self.dnn_layers,
                              drop_rate=args.drop_rate)

        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):

        input_embs_id = self.id_embedding(sample_items_id)
        if self.use_modal:
            input_embs_all = self.mlp_layers(self.fc(input_embs_id, self.turn_dim(input_embs_content)))
        else:
            input_embs_all = input_embs_id
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        return loss
