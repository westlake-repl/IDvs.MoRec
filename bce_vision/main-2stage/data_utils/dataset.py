import torch
from torch.utils.data import Dataset
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
import random


class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)



class BuildTrainDataset(Dataset):
    def __init__(self, user_history, train_pairs, item_content,
                 item_num, max_seq_len, use_modal):
        self.user_history = user_history
        self.train_pairs = train_pairs
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal
        self.neg_num = 1
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
        return torch.LongTensor([user_id]), sample_items, self.bce_label


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


class Build_Id_Eval_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class Build_Lmdb_Eval_Dataset(Dataset):
    def __init__(self, data, item_id_to_keys, db_path, resize):
        self.data = data
        self.item_id_to_keys = item_id_to_keys
        self.db_path = db_path
        self.resize = resize
        self.padding_emb = torch.zeros((3, self.resize, self.resize))
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transforms.Compose([
                tv.transforms.Resize((self.resize, self.resize)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item_id = self.data[index]
        if index == 0:
            return self.padding_emb
        env = self.env
        with env.begin() as txn:
            byteflow = txn.get(self.item_id_to_keys[item_id])
        IMAGE = pickle.loads(byteflow)
        img = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
        return torch.FloatTensor(img)



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
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
