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


class Build_Id_Dataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, neg_sampling_list):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.neg_sampling_list = neg_sampling_list
        self.inter_max_index = len(self.neg_sampling_list)-1

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len + [1] * tokens_Len
        sample_items = torch.LongTensor([0] * mask_len + seq)

        sample_items_id = sample_items
        return sample_items_id, sample_items, torch.FloatTensor(log_mask)


class Build_Lmdb_Dataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, db_path, item_id_to_keys, resize, neg_sampling_list):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.neg_sampling_list = neg_sampling_list
        self.inter_max_index = len(self.neg_sampling_list)-1
        self.db_path = db_path
        self.item_id_to_keys = item_id_to_keys
        self.resize = resize
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
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = len(seq) - 1
        mask_len = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len + [1] * tokens_Len

        sample_items_id = torch.LongTensor([0] * mask_len + seq)
        sample_items = np.zeros((self.max_seq_len, 3, self.resize, self.resize))

        env = self.env
        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]]))
                image_trans = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
                sample_items[mask_len + i] = image_trans
            # target
            IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]]))
            image_trans = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
            sample_items[mask_len + tokens_Len] = image_trans
        return sample_items_id, torch.FloatTensor(sample_items), torch.FloatTensor(log_mask)


class BuildEvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels


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
