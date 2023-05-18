import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BeitForImageClassification, SwinForImageClassification, CLIPVisionModel, ViTMAEModel

from parameters import parse_args
from model import Model, Resnet_Encoder, Vit_Encoder
from data_utils import read_images, read_behaviors, Build_Lmdb_Dataset, Build_Id_Dataset, LMDB_Image, \
    eval_model, get_itemId_embeddings, get_itemLMDB_embeddings, get_image_embs
from data_utils.utils import *
import torchvision.models as models
from torch import nn
import random
from torch.cuda.amp import autocast

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_, constant_

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(args, use_modal, local_rank):
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
        elif 'swin' in args.CV_model_load:
            cv_model_load = '../../pretrained_models/' + args.CV_model_load
            cv_model = SwinForImageClassification.from_pretrained(cv_model_load)
            num_fc_ftr = cv_model.classifier.in_features
        else:
            cv_model = None
            num_fc_ftr = None
        cv_model = torch.nn.Sequential(*(list(cv_model.children())[:-1]))

        for index, (name, param) in enumerate(cv_model.named_parameters()):
            param.requires_grad = False


    else:
        cv_model = None

    Log_file.info('read images...')
    before_item_id_to_keys, before_item_name_to_id, before_item_id_to_name = read_images(
        os.path.join(args.root_data_dir, args.dataset, args.images))
    Log_file.info('read behaviors...')
    item_num, item_id_to_keys, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, item_name_to_id = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_keys,
                       before_item_name_to_id, before_item_id_to_name, args.max_seq_len, args.min_seq_len, Log_file)

    Log_file.info('build dataset...')
    if use_modal:

        Log_file.info('CV Encoder...')

        if 'resnet' in args.CV_model_load:
            cv_encoder = Resnet_Encoder(image_net=cv_model).to(local_rank)
        else:
            cv_encoder = Vit_Encoder(image_net=cv_model).to(local_rank)

        Log_file.info('get output...')
        item_image_embs = get_image_embs(cv_encoder, item_num, item_id_to_keys,
                                         os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                         256, args, local_rank)
        Log_file.info('item_image_embs...')
        Log_file.info(item_image_embs.shape)

        train_dataset = Build_Lmdb_Dataset(u2seq=users_train, item_num=item_num, max_seq_len=args.max_seq_len,
                                           item_content=item_image_embs)
    else:
        train_dataset = Build_Id_Dataset(u2seq=users_train, item_num=item_num, max_seq_len=args.max_seq_len)

    Log_file.info('build model...')
    model = Model(args, item_num, num_fc_ftr, use_modal).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    if use_modal:
        Log_file.info(model.turn_dim)
        Log_file.info(model.fc)
        Log_file.info(model.mlp_layers)

    if 'None' not in args.load_ckpt_name:
        Log_file.info('load ckpt if not None...')
        ckpt_path = get_checkpoint(model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'])
        Log_file.info(f"Model loaded from {ckpt_path}")
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        is_early_stop = False
    else:
        is_early_stop = True
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    run_eval(model, item_image_embs, args.lmdb_data, users_history_for_test, users_test, 256, item_num, use_modal,
                         args.mode, is_early_stop, local_rank)

def run_eval(model, item_image_embs, lmdb_path, user_history, users_eval, batch_size, item_num, use_modal,
             mode, is_early_stop, local_rank):
    Log_file.info('Validating...')
    if use_modal:
        item_embeddings = get_itemLMDB_embeddings(
            model, item_image_embs, batch_size, args, use_modal, local_rank)
    else:
        item_embeddings = get_itemId_embeddings(model, item_num, batch_size, args, local_rank)
    valid_Hit10 = eval_model(model, user_history, users_eval, item_embeddings, batch_size, args,
                             item_num, Log_file, mode, local_rank)
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
        dir_label = args.behaviors + f' _{model_load}_MOdnn_{args.mo_dnn_layers}'

        log_paras = f'{args.item_tower}-{model_load}_MOdnn_{args.mo_dnn_layers}_dnn_{args.dnn_layers}' \
                    f'_ed_{args.embedding_dim}_bs_{args.batch_size*gpus}' \
                    f'_lr_{args.lr}_Flr_{args.fine_tune_lr}' \
                    f'_L2_{args.l2_weight}'
    else:
        is_use_modal = False
        model_load = 'id'
        dir_label = str(args.item_tower) + ' ' + args.behaviors
        log_paras = f'{model_load}' \
                    f'_ed_{args.embedding_dim}_bs_{args.batch_size}' \
                    f'_lr_{args.lr}_L2_{args.l2_weight}'

    model_dir = os.path.join('./checkpoint_' + dir_label, 'cpt_' + log_paras)
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run

    Log_file, Log_screen = setuplogger('test', args.behaviors + ' ' + log_paras,
                                       time_run, args.mode, dist.get_rank(), args.behaviors)

    Log_file.info(args)
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if 'train' in args.mode:
        train(args, is_use_modal, local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
