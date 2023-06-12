import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .dataset import BuildEvalDataset, SequentialDistributedSampler
import torch.distributed as dist
import math


class ItersDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def id_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr


def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t+"_results   {}".format('\t'.join(["{:0.5f}".format(i * 100) for i in x])))


def get_mean(arr):
    return [i.mean() for i in arr]


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = distributed_concat(eval_m, len(test_sampler.dataset))\
            .to(torch.device("cpu")).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result


def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return eval_ra


def get_user_embeddings(model, user_num, test_batch_size, args, local_rank):
    model.eval()
    user_dataset = ItersDataset(data=np.arange(user_num + 1))
    user_dataloader = DataLoader(user_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=id_collate_fn)
    user_embeddings = []
    with torch.no_grad():
        for input_ids in user_dataloader:
            input_ids = input_ids.to(local_rank)
            user_emb = model.module.user_embedding(input_ids)
            user_feature = model.module.user_encoder(user_emb)
            user_embeddings.extend(user_feature)
    return torch.stack(tensors=user_embeddings, dim=0).to(torch.device("cpu")).detach()


def get_item_embeddings(model, item_content, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = ItersDataset(data=item_content)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=id_collate_fn)
    item_embeddings = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            if use_modal:
                item_feature = model.module.bert_encoder(input_ids)
            else:
                item_emb = model.module.id_embedding(input_ids)
                item_feature = model.module.id_encoder(item_emb)
            item_embeddings.extend(item_feature)
    return torch.stack(tensors=item_embeddings, dim=0).detach()


def eval_model(model, user_history, eval_pairs, user_embeddings, item_embeddings, test_batch_size,
               args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(eval_pairs=eval_pairs, user_content=user_embeddings, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, user_embs, labels = data
            user_ids, user_embs, labels = \
                user_ids.to(local_rank), user_embs.to(local_rank), labels.to(local_rank)
            scores = torch.matmul(user_embs, item_embeddings.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]
