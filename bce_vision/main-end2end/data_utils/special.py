import torch
from torch.utils.data import Dataset, DataLoader
from .dataset import BuildEvalDataset, Build_Lmdb_Eval_Dataset, Build_Id_Eval_Dataset, SequentialDistributedSampler
import torch.distributed as dist
import math
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision as tv
import torchvision.transforms as transforms
import lmdb
import pickle
import os


def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return eval_ra


def get_mean(arr):
    return [i.mean() for i in arr]


def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t+"_results   {}".format('\t'.join(["{:0.5f}".format(i * 100) for i in x])))


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


def read_behaviors_special(file_for_cold_seq, file_for_new_seq, file_for_new_items, item_name_to_id, Log_file, use_modal):
    user_id_for_cold = 0
    seqs_for_cold = {}
    history_for_cold = {}
    cold_count = set()
    Log_file.info('rebuild file_for_cold_seq...')
    with open(file_for_cold_seq, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            history_item_name = splited[1].split(' ')
            seqs_input_and_target = [item_name_to_id[i] for i in history_item_name]
            seqs_for_cold[user_id_for_cold] = seqs_input_and_target
            cold_count.add(seqs_input_and_target[-1])
            history_for_cold[user_id_for_cold] = torch.LongTensor(np.array(seqs_input_and_target[:-1]))
            user_id_for_cold += 1
    Log_file.info('seqs_for_cold: {}'.format(len(seqs_for_cold)))
    Log_file.info('cold_count for eval: {}'.format(len(cold_count)))

    if use_modal:
        new_item_id_to_keys = {}
        new_item_name_to_id = {}
        new_item_id = 1
        with open(file_for_new_items, "r") as f:
            for line in f:
                splited = line.strip('\n').split('\t')
                image_name = splited[0]
                new_item_name_to_id[image_name] = new_item_id
                new_item_id_to_keys[new_item_id] = u'{}'.format(int(image_name.replace('v', ''))).encode('ascii')
                new_item_id += 1

        user_id_for_new = 0
        seqs_for_new = {}
        history_for_new = {}
        Log_file.info('rebuild file_for_new_seq...')
        with open(file_for_new_seq, "r") as f:
            for line in f:
                splited = line.strip('\n').split('\t')
                history_item_name = splited[1].split(' ')
                new_item_name = history_item_name[-1]
                target_id = new_item_name_to_id[new_item_name]
                seqs_input = [item_name_to_id[i] for i in history_item_name[:-1]]
                seqs_for_new[user_id_for_new] = seqs_input + [target_id]
                history_for_new[user_id_for_new] = torch.LongTensor(np.array(seqs_input))
                user_id_for_new += 1
        Log_file.info('seqs_for_new: {}'.format(len(seqs_for_new)))
        Log_file.info('new_item_id_to_keys: {}'.format(len(new_item_id_to_keys)))
        return seqs_for_cold, history_for_cold, seqs_for_new, history_for_new, new_item_id_to_keys
    else:
        return seqs_for_cold, history_for_cold, None, None, None


class BuildEvalColdDataset(Dataset):
    def __init__(self, seqs_for_cold, item_embeddings, max_seq_len, item_num):
        self.seqs_for_cold = seqs_for_cold
        self.item_embeddings = item_embeddings
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num

    def __len__(self):
        return len(self.seqs_for_cold)

    def __getitem__(self, user_id):
        seq = self.seqs_for_cold[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_embeddings[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels


class BuildEvalNewDataset(Dataset):
    def __init__(self, seqs_for_new, item_embeddings, new_item_embeddings, max_seq_len, item_num):
        self.seqs_for_new = seqs_for_new
        self.item_embeddings = item_embeddings
        self.new_item_embeddings = new_item_embeddings
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num

    def __len__(self):
        return len(self.seqs_for_new)

    def __getitem__(self, user_id):
        seq = self.seqs_for_new[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        target_embedding = self.new_item_embeddings[target]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_embeddings[pad_tokens]
        labels = np.zeros(self.item_num + 1)
        labels[-1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels, target_embedding


def eval_model_special(model, item_embeddings, new_item_embeddings, test_batch_size, args, item_num,
                       Log_file, seqs_for_cold, history_for_cold, seqs_for_new, history_for_new, local_rank, use_modal):
    eval_dataset = BuildEvalColdDataset(seqs_for_cold=seqs_for_cold, item_embeddings=item_embeddings,
                                        max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info("cold_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = history_for_cold[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, 'cold')
    if use_modal:
        item_embeddings = item_embeddings.to(torch.device("cpu")).detach()
        eval_dataset = BuildEvalNewDataset(seqs_for_new=seqs_for_new, item_embeddings=item_embeddings,
                                           new_item_embeddings=new_item_embeddings, max_seq_len=args.max_seq_len,
                                           item_num=item_num)
        test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
        eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                             num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        model.eval()
        topK = 10
        Log_file.info("new_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
        item_embeddings = item_embeddings.to(local_rank)
        with torch.no_grad():
            eval_all_user = []
            item_rank = torch.Tensor(np.arange(item_num+1) + 1).to(local_rank)
            for data in eval_dl:
                user_ids, input_embs, log_mask, labels, target_embeddings = data
                user_ids, input_embs, log_mask, labels, target_embeddings = \
                    user_ids.to(local_rank), input_embs.to(local_rank), \
                    log_mask.to(local_rank), labels.to(local_rank).detach(), target_embeddings.to(local_rank)
                prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
                for user_id, label, user_prec_emb, target_emb in zip(user_ids, labels, prec_emb, target_embeddings):
                    item_emb = torch.cat((item_embeddings, target_emb.unsqueeze(0)), 0)
                    score = torch.matmul(user_prec_emb, item_emb.t()).squeeze(dim=-1).detach()
                    user_id = user_id[0].item()
                    history = history_for_new[user_id].to(local_rank)
                    score[history] = -np.inf
                    score = score[1:]
                    eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
            eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
            Hit10, nDCG10 = eval_all_user
            mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
            print_metrics(mean_eval, Log_file, 'new')

