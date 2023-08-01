import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BeitForImageClassification, SwinForImageClassification, CLIPVisionModel, ViTMAEModel

from parameters import parse_args
from model import Model
from data_utils import read_images, read_behaviors, Build_Lmdb_Dataset, Build_Id_Dataset, LMDB_Image, \
    eval_model, get_itemId_embeddings, get_itemLMDB_embeddings, \
    read_behaviors_special, eval_model_special
from data_utils.utils import *
import torchvision.models as models
from torch import nn
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_, constant_

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test(args, use_modal, local_rank):
    if use_modal:
        if 'resnet' in args.CV_model_load:
            cv_model_load = '../../pretrained_models/' + args.CV_model_load
            if '18' in cv_model_load:
                cv_model = models.resnet18(pretrained=False)
            elif '34' in cv_model_load:
                cv_model = models.resnet34(pretrained=False)
            elif '50' in cv_model_load:
                cv_model = models.resnet50(pretrained=False)
            elif '101' in cv_model_load:
                cv_model = models.resnet101(pretrained=False)
            elif '152' in cv_model_load:
                cv_model = models.resnet152(pretrained=False)
            else:
                cv_model = None
            cv_model.load_state_dict(torch.load(cv_model_load))
            num_fc_ftr = cv_model.fc.in_features
            cv_model.fc = nn.Linear(num_fc_ftr, args.embedding_dim)
            xavier_normal_(cv_model.fc.weight.data)
            if cv_model.fc.bias is not None:
                constant_(cv_model.fc.bias.data, 0)
        elif 'swin' in args.CV_model_load:
            cv_model_load = '../../pretrained_models/' + args.CV_model_load
            cv_model = SwinForImageClassification.from_pretrained(cv_model_load)
            num_fc_ftr = cv_model.classifier.in_features
            cv_model.classifier = nn.Linear(num_fc_ftr, args.embedding_dim)
            xavier_normal_(cv_model.classifier.weight.data)
            if cv_model.classifier.bias is not None:
                constant_(cv_model.classifier.bias.data, 0)
        else:
            cv_model = None

        for index, (name, param) in enumerate(cv_model.named_parameters()):
            if index < args.freeze_paras_before:
                param.requires_grad = False
    else:
        cv_model = None

    Log_file.info('read images...')
    before_item_id_to_keys, before_item_name_to_id, before_item_id_to_name = read_images(
        os.path.join(args.root_data_dir, args.dataset, args.images))
    Log_file.info('read behaviors...')
    item_num, item_id_to_keys, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, item_name_to_id, neg_sampling_list, pop_prob_list = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_keys,
                       before_item_name_to_id, before_item_id_to_name, args.max_seq_len, args.min_seq_len, Log_file)



    Log_file.info('build model...')
    model = Model(args, item_num, use_modal, cv_model, pop_prob_list).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    Log_file.info('load ckpt if not None...')
    Log_file.info(os.path.join(model_dir, args.load_ckpt_name))
    ckpt_path = get_checkpoint(model_dir, args.load_ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    Log_file.info('load checkpoint...')
    model.load_state_dict(checkpoint['model_state_dict'])
    Log_file.info(f"Model loaded from {ckpt_path}")
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    Log_file.info('model.cuda()...')
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_num = sum(p.numel() for p in model.module.parameters())
    Log_file.info("##### total_num {} #####".format(total_num))

    run_eval_test(model, item_id_to_keys, args.lmdb_data, users_history_for_test, users_test, 256, item_num,
                  use_modal, args.mode, local_rank)


def run_eval_test(model, item_id_to_keys, lmdb_path, user_history, users_eval, batch_size, item_num, use_modal,
                  mode, local_rank):
    eval_start_time = time.time()
    if use_modal:
        item_embeddings = get_itemLMDB_embeddings(model, item_num, item_id_to_keys, lmdb_path, batch_size, args, local_rank)
    else:
        item_embeddings = get_itemId_embeddings(model, item_num, batch_size, args, local_rank)
    eval_model(model, user_history, users_eval, item_embeddings, batch_size, args,
               item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    setup_seed(12345)
    gpus = torch.cuda.device_count()

    if 'modal' in args.item_tower:
        is_use_modal = True
        model_load = args.CV_model_load.replace('.pth', '')
        dir_label = str(args.item_tower) + f'_{model_load}_freeze_{args.freeze_paras_before}'

        log_paras = f'{model_load}-{args.freeze_paras_before}' \
                    f'_ed_{args.embedding_dim}_bs_{args.batch_size*gpus}' \
                    f'_lr_{args.lr}_Flr_{args.fine_tune_lr}' \
                    f'_L2_{args.l2_weight}_FL2_{args.fine_tune_l2_weight}'
    else:
        is_use_modal = False
        model_load = 'id'
        dir_label = str(args.item_tower)
        log_paras = f'{model_load}' \
                    f'_ed_{args.embedding_dim}_bs_{args.batch_size}' \
                    f'_lr_{args.lr}_Flr_{args.fine_tune_lr}' \
                    f'_L2_{args.l2_weight}_FL2_{args.fine_tune_l2_weight}'

    model_dir = os.path.join('./checkpoint_' + dir_label, 'cpt_' + log_paras)
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())

    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank(), args.behaviors)
    Log_file.info(args)

    test(args, is_use_modal, local_rank)
