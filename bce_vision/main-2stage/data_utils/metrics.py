import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import BuildEvalDataset, Build_Lmdb_Eval_Dataset, \
    Build_Id_Eval_Dataset, SequentialDistributedSampler
import torch.distributed as dist
from torch.utils.data import Dataset
import os
import math
import tqdm


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


def get_image_embs(cv_encoder, item_num, item_id_to_keys, lmdb_data, test_batch_size, args, local_rank):
    cv_encoder.eval()
    item_dataset = Build_Lmdb_Eval_Dataset(data=np.arange(item_num + 1), item_id_to_keys=item_id_to_keys,
                                           db_path=os.path.join(args.root_data_dir, args.dataset, lmdb_data),
                                           resize=args.CV_resize)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                 num_workers=args.num_workers, pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = cv_encoder(input_ids).to(torch.device("cpu")).detach()
            item_emb = torch.flatten(item_emb, 1)
            item_embeddings.extend(item_emb)
    return torch.stack(tensors=item_embeddings, dim=0)


class BuildItemEmbeddingDataset(Dataset):
    def __init__(self, item_word_embs):
        self.item_word_embs = item_word_embs

    def __len__(self):
        return len(self.item_word_embs)

    def __getitem__(self, item_id):
        return torch.FloatTensor(self.item_word_embs[item_id])

def get_user_embeddings(model, user_num, test_batch_size, args, local_rank):
    model.eval()
    user_dataset = Build_Id_Eval_Dataset(data=np.arange(user_num + 1))
    user_dataloader = DataLoader(user_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=id_collate_fn)
    user_embeddings = []
    with torch.no_grad():
        for input_ids in user_dataloader:
            input_ids = input_ids.to(local_rank)
            user_emb = model.module.user_embedding(input_ids)
            user_embeddings.extend(user_emb)
    return torch.stack(tensors=user_embeddings, dim=0).to(torch.device("cpu")).detach()


def get_item_embeddings(model, item_image_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_image_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_embs_content in item_dataloader:
            input_embs_content = input_embs_content.to(local_rank)
            input_embs_all = model.module.fc(model.module.turn_dim(input_embs_content))\
                .to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)


def eval_model(model, user_history, eval_pairs, user_embeddings, item_embeddings,
               test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(eval_pairs=eval_pairs, user_content=user_embeddings, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
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

