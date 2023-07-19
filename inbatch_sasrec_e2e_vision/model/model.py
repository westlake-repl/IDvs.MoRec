import torch
from torch import nn
from torch.nn.init import xavier_normal_
from .encoders import Resnet_Encoder, Vit_Encoder, User_Encoder, MAE_Encoder


class Model(torch.nn.Module):
    def __init__(self, args, item_num, use_modal, image_net, pop_prob_list):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)

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
        else:
            self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, sample_items_id, sample_items, log_mask, local_rank):
        self.pop_prob_list = self.pop_prob_list.to(local_rank)
        debias_logits = torch.log(self.pop_prob_list[sample_items_id.view(-1)])
        if self.use_modal:
            score_embs = self.cv_encoder(sample_items)
        else:
            score_embs = self.id_embedding(sample_items)

        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)

        prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        prec_vec = prec_vec.view(-1, self.args.embedding_dim)  # (bs*max_seq_len, ed)

        ######################################  IN-BATCH CROSS-ENTROPY LOSS  ######################################
        bs = log_mask.size(0)
        ce_label = torch.tensor(
            [i * self.max_seq_len + i + j for i in range(bs) for j in range(1, self.max_seq_len + 1)],
            dtype=torch.long).to(local_rank)
        logits = torch.matmul(prec_vec, score_embs.t())  # (batch_size*max_seq_len, batch_size*(max_seq_len+1))
        logits = logits - debias_logits
        logits[:, torch.cat((log_mask, torch.ones(log_mask.size(0))
                             .unsqueeze(-1).to(local_rank)), dim=1).view(-1) == 0] = -1e4
        logits = logits.view(bs, self.max_seq_len, -1)
        id_list = sample_items_id.view(bs, -1)  # sample_items_id (bs, max_seq_len)
        for i in range(bs):
            reject_list = id_list[i]  # reject_list (max_seq_len)
            u_ids = sample_items_id.repeat(self.max_seq_len).expand((len(reject_list), -1))
            reject_mat = reject_list.expand((u_ids.size(1), len(reject_list))).t()
            # (max_seq_len, batch_size*(max_seq_len+1))
            mask_mat = (u_ids == reject_mat).any(axis=0).reshape(logits[i].shape)
            for j in range(self.max_seq_len):
                mask_mat[j][i * (self.max_seq_len + 1) + j + 1] = False
            logits[i][mask_mat] = -1e4

        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * self.max_seq_len, -1)
        loss = self.criterion(logits[indices], ce_label[indices])

        return loss
