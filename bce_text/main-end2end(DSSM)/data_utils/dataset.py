import torch
from torch.utils.data import Dataset
import numpy as np
import random
import torch.distributed as dist
import math


class BuildTrainDataset(Dataset):
    def __init__(self, user_history, train_pairs, item_num, item_content, neg_num, use_modal):
        self.user_history = user_history
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.item_content = item_content
        self.use_modal = use_modal
        self.neg_num = neg_num
        self.bce_label = torch.FloatTensor(np.array([1] + [0]*self.neg_num))

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):
        (user_id, pos_id) = self.train_pairs[index]
        history = self.user_history[user_id]
        neg_items = []
        for i in range(self.neg_num):
            neg_id = random.randint(1, self.item_num)
            while neg_id in history:
                neg_id = random.randint(1, self.item_num)
            neg_items.append(neg_id)
        sample_items = [pos_id] + neg_items
        if self.use_modal:
            sample_items = self.item_content[sample_items]
        return torch.LongTensor([user_id]), torch.LongTensor(sample_items), self.bce_label


class BuildEvalDataset(Dataset):
    def __init__(self, eval_pairs, user_content, item_num):
        self.eval_pairs = eval_pairs
        self.user_content = user_content
        self.item_num = item_num

    def __len__(self):
        return len(self.eval_pairs)

    def __getitem__(self, index):
        (user_id, target) = self.eval_pairs[index]
        user_emb = self.user_content[user_id]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), user_emb, labels


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices += [indices[-1]] * (self.total_size - len(indices))
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
