import torch
from torch import nn
from torch.nn.init import xavier_normal_
from .encoders import Resnet_Encoder, Vit_Encoder, \
    User_Encoder, MAE_Encoder, ADD, CAT


class Model(torch.nn.Module):
    def __init__(self, args, item_num, use_modal, image_net):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len

        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        if self.use_modal:
            if 'resnet' in args.CV_model_load:
                self.cv_encoder = Resnet_Encoder(image_net=image_net)
            elif 'beit' in args.CV_model_load or 'swin' in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net)
            elif 'mae' in args.CV_model_load or "checkpoint" in args.CV_model_load:
                self.cv_encoder = MAE_Encoder(image_net=image_net, item_dim=args.embedding_dim)

        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            if 'cat_3' in args.item_tower:
                dnn_layer = 3
            else:
                dnn_layer = 1
            self.fc = CAT(layers=[args.embedding_dim*2] + [args.embedding_dim] * dnn_layer,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_items_id, sample_items_content, log_mask, local_rank):
        input_embs_id = self.id_embedding(sample_items_id)
        if self.use_modal:
            input_embs_content = self.cv_encoder(sample_items_content)
            input_embs_all = self.fc(input_embs_id, input_embs_content)
        else:
            input_embs_all = input_embs_id

        input_embs = input_embs_all.view(-1, self.max_seq_len + 1, 2, self.args.embedding_dim)

        prec_vec = self.user_encoder(input_embs[:, :, 0][:, :-1, :], log_mask, local_rank)
        pos_score = (prec_vec * input_embs[:, :, 0][:, 1:, :]).sum(-1)
        neg_score = (prec_vec * input_embs[:, :, 1][:, :-1, :]).sum(-1)

        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        return loss

