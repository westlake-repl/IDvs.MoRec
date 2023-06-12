import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_
from .encoders import Resnet_Encoder, Vit_Encoder, MLP_Encoder


class Model(torch.nn.Module):
    def __init__(self, args, user_num, item_num, use_modal, image_net):
        super(Model, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.embedding_dim = args.embedding_dim
        self.neg_num = args.neg_num

        self.user_embedding = nn.Embedding(user_num + 1, self.embedding_dim, padding_idx=0)
        xavier_normal_(self.user_embedding.weight.data)
        self.user_encoder = MLP_Encoder(embedding_dim=self.embedding_dim,
                                        dnn_layers=args.dnn_layers,
                                        drop_rate=args.drop_rate)

        if self.use_modal:
            if 'resnet' in args.CV_model_load:
                self.cv_encoder = Resnet_Encoder(image_net=image_net)
            elif 'beit' in args.CV_model_load or 'swin' in args.CV_model_load:
                self.cv_encoder = Vit_Encoder(image_net=image_net)
        else:
            self.id_embedding = nn.Embedding(item_num + 1, self.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_embedding.weight.data)
            self.id_encoder = MLP_Encoder(embedding_dim=self.embedding_dim,
                                          dnn_layers=args.dnn_layers,
                                          drop_rate=args.drop_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_user, sample_items, bce_label):
        user_embedding = self.user_embedding(input_user)
        user_feature = self.user_encoder(user_embedding)
        if self.use_modal:
            item_feature = self.cv_encoder(sample_items)
        else:
            item_embedding = self.id_embedding(sample_items)
            item_feature = self.id_encoder(item_embedding)
        item_feature = item_feature.view(-1, 1 + self.neg_num, self.embedding_dim)
        score = torch.bmm(item_feature, user_feature.unsqueeze(-1)).squeeze(dim=-1)
        loss = self.criterion(score.view(-1), bce_label.view(-1))
        return loss
