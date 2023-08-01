import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_
from .encoders import Resnet_Encoder, User_Encoder, MLP_Layers,Swin_Encoder


class Model(torch.nn.Module):
    def __init__(self, args, item_num, num_fc_ftr):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len

        self.fc = MLP_Layers(word_embedding_dim=num_fc_ftr,
                             item_embedding_dim=args.embedding_dim,
                             layers=[args.embedding_dim] * (args.dnn_layer + 1),
                             drop_rate=args.drop_rate)

        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_items, log_mask, local_rank):
        input_embs_all = self.fc(sample_items)

        input_embs = input_embs_all.view(-1, self.max_seq_len + 1, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)

        pos_labels, neg_labels = torch.ones(pos_score.shape), torch.zeros(neg_score.shape)
        # pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        return loss
