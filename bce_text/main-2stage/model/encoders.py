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


class Text_Encoder_mean(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder_mean, self).__init__()
        self.bert_model = bert_model
        # self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        # self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        input_mask_expanded = text_attmask.unsqueeze(-1).expand(hidden_states.size()).float()
        mean_output = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return mean_output
        # mean_output = self.fc(mean_output)
        # return self.activate(mean_output)


class Text_Encoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder, self).__init__()
        self.bert_model = bert_model
        # self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        # self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        return hidden_states[:, 0]
        # cls = self.fc(hidden_states[:, 0])
        # return self.activate(cls)


class Bert_Encoder(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(Bert_Encoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 2,
            'abstract': args.num_words_abstract * 2,
            'body': args.num_words_body * 2
        }
        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)]
            )
            for key in self.attributes2length.keys()
        }

        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'abstract', 'body']

        if 'opt' in args.bert_model_load:
            self.text_encoders = nn.ModuleDict({
                'title': Text_Encoder_mean(bert_model, args.embedding_dim, args.word_embedding_dim)
            })
        else:
            self.text_encoders = nn.ModuleDict({
                    'title': Text_Encoder(bert_model, args.embedding_dim, args.word_embedding_dim)
            })

        self.newsname = [name for name in set(args.news_attributes) & set(text_encoders_candidates)]

    def forward(self, news):
        text_vectors = [
            self.text_encoders['title'](
                torch.narrow(news, 1, self.attributes2start[name], self.attributes2length[name]))
            for name in self.newsname
        ]
        if len(text_vectors) == 1:
            final_news_vector = text_vectors[0]
        else:
            final_news_vector = torch.mean(torch.stack(text_vectors, dim=1), dim=1)
        return final_news_vector
