import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_


class MLP_Layers(torch.nn.Module):
    def __init__(self, layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=drop_rate))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.GELU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

    def forward(self, x):
        return self.mlp_layers(x)


class MLP_Encoder(torch.nn.Module):
    def __init__(self, embedding_dim, dnn_layers, drop_rate):
        super(MLP_Encoder, self).__init__()
        self.dnn_layers = dnn_layers
        if self.dnn_layers > 0:
            self.MLP = MLP_Layers(layers=[embedding_dim] * (self.dnn_layers + 1), drop_rate=drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, inputs):
        if self.dnn_layers > 0:
            return self.MLP(inputs)
        else:
            return inputs


class Text_Encoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        cls = self.fc(hidden_states[:, 0])
        return self.activate(cls)


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
