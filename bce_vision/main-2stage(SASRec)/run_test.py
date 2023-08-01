import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BeitForImageClassification, SwinForImageClassification, CLIPVisionModel

from parameters import parse_args
from model import Model, Resnet_Encoder
from data_utils import read_images, read_behaviors, BuildTrainDataset, LMDB_Image, \
    eval_model, get_item_embeddings, get_image_embs
from data_utils.utils import *
import torchvision.models as models
from torch import nn
import random
from torch.cuda.amp import autocast

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_, constant_

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test(args, local_rank):
    if 'resnet' in args.CV_model_load:
        cv_model_load = '/yuanzheng/id_modal/pretrained_models/resnet/' + args.CV_model_load
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
    else:
        cv_model_load = '/yuanzheng/id_modal/pretrained_models/swin/' + args.CV_model_load
        cv_model = SwinForImageClassification.from_pretrained(cv_model_load)
        num_fc_ftr = cv_model.classifier.in_features
    cv_model = torch.nn.Sequential(*(list(cv_model.children())[:-1]))

    for index, (name, param) in enumerate(cv_model.named_parameters()):
        param.requires_grad = False

    Log_file.info('read images...')
    before_item_id_to_keys, before_item_name_to_id = read_images(
        os.path.join(args.root_data_dir, args.dataset, args.images))

    Log_file.info('read behaviors...')
    item_num, item_id_to_keys, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, user_name_to_id = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_keys,
                       before_item_name_to_id, args.max_seq_len, args.min_seq_len, Log_file)

    Log_file.info('CV Encoder...')
    cv_encoder = Resnet_Encoder(image_net=cv_model).to(local_rank)

    Log_file.info('get resnet output...')
    item_image_embs = get_image_embs(cv_encoder, item_num, item_id_to_keys,
                                     os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                     256, args, local_rank)

    Log_file.info('build model...')
    model = Model(args, item_num, num_fc_ftr).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    Log_file.info('load ckpt if not None...')
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
    trainable_num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    Log_file.info('\n')
    Log_file.info('Testing...')
    run_eval_test(model, item_image_embs, users_history_for_test, users_test, 256, item_num,
             args.mode, local_rank)


def run_eval_test(model, item_image_embs, user_history, users_eval, batch_size, item_num,
             mode, local_rank):
    eval_start_time = time.time()
    item_embeddings = get_item_embeddings(model, item_image_embs, batch_size, args, local_rank)
    eval_model(model, user_history, users_eval, item_embeddings, batch_size, args,
                             item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)

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

    model_load = args.CV_model_load.replace('.pth', '')
    dir_label = str(args.item_tower) + f'_{model_load}'

    if args.dnn_layer==0:
        log_paras = f'{args.item_tower}_{model_load}_bs_{args.batch_size}'\
                    f'_ed_{args.embedding_dim}_lr_{args.lr}'\
                    f'_L2_{args.l2_weight}_dp_{args.drop_rate}_Flr_{args.fine_tune_lr}'
    else:
        log_paras = f'{args.item_tower}_{model_load}_dnn_{args.dnn_layer}_bs_{args.batch_size}' \
                    f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                    f'_L2_{args.l2_weight}_dp_{args.drop_rate}_Flr_{args.fine_tune_lr}'
    model_dir = os.path.join('./checkpoint_' + dir_label, 'cpt_' + log_paras)
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run

    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank(), args.behaviors)

    Log_file.info(args)
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if 'test' in args.mode:
        test(args, local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
