import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, \
    RobertaTokenizer, RobertaModel, RobertaConfig, \
    DebertaTokenizer, DebertaModel, DebertaConfig, \
    DistilBertTokenizer, DistilBertModel, DistilBertConfig, \
    GPT2Tokenizer, OPTModel, OPTConfig

from parameters import parse_args
from model import Model
from data_utils import read_news, read_news_bert, get_doc_input_bert, \
    read_behaviors, BuildTrainDataset, eval_model, get_bert_embeddings, get_id_embeddings
from data_utils.utils import *
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(args, use_modal, local_rank):
    if use_modal:
        if 'roberta' in args.bert_model_load:
            Log_file.info('load roberta model...')
            bert_model_load = '../../pretrained_models/' + args.bert_model_load
            tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
            config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
            if 'base' in args.bert_model_load:
                pooler_para = [197, 198]
                args.word_embedding_dim = 768
            if 'large' in args.bert_model_load:
                pooler_para = [389, 390]
                args.word_embedding_dim = 1024
        elif 'opt' in args.bert_model_load:
            Log_file.info('load opt model...')
            bert_model_load = '../../pretrained_models/' + args.bert_model_load
            tokenizer = GPT2Tokenizer.from_pretrained(bert_model_load)
            config = OPTConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = OPTModel.from_pretrained(bert_model_load, config=config)
            pooler_para = []
        else:
            Log_file.info('load bert model...')
            bert_model_load = '../../pretrained_models/' + args.bert_model_load
            tokenizer = BertTokenizer.from_pretrained(bert_model_load)
            config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = BertModel.from_pretrained(bert_model_load, config=config)

            if 'tiny' in args.bert_model_load:
                pooler_para = [37, 38]
                args.word_embedding_dim = 128
            if 'mini' in args.bert_model_load:
                pooler_para = [69, 70]
                args.word_embedding_dim = 256
            if 'small' in args.bert_model_load:
                pooler_para = [69, 70]
                args.word_embedding_dim = 512
            if 'medium' in args.bert_model_load:
                pooler_para = [133, 134]
                args.word_embedding_dim = 512
            if 'base' in args.bert_model_load:
                pooler_para = [197, 198]
                args.word_embedding_dim = 768
            if 'large' in args.bert_model_load:
                pooler_para = [389, 390]
                args.word_embedding_dim = 1024
        for index, (name, param) in enumerate(bert_model.named_parameters()):
            if index < args.freeze_paras_before or index in pooler_para:
                param.requires_grad = False

        Log_file.info('read news...')
        before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_news_bert(
            os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)

        Log_file.info('read behaviors...')
        item_num, item_id_to_dic, users_train, users_valid, users_test, \
        users_history_for_valid, users_history_for_test, item_name_to_id = \
            read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_dic,
                           before_item_name_to_id, before_item_id_to_name,
                           args.max_seq_len, args.min_seq_len, Log_file)

        Log_file.info('combine news information...')
        news_title, news_title_attmask, \
        news_abstract, news_abstract_attmask, \
        news_body, news_body_attmask = get_doc_input_bert(item_id_to_dic, args)

        item_content = np.concatenate([
            x for x in
            [news_title, news_title_attmask,
             news_abstract, news_abstract_attmask,
             news_body, news_body_attmask]
            if x is not None], axis=1)
    else:
        before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_news(os.path.join(args.root_data_dir, args.dataset, args.news))

        Log_file.info('read behaviors...')
        item_num, item_id_to_dic, users_train, users_valid, users_test, \
        users_history_for_valid, users_history_for_test, item_name_to_id = \
            read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_dic,
                           before_item_name_to_id, before_item_id_to_name, args.max_seq_len, args.min_seq_len, Log_file)
        item_content = np.arange(item_num + 1)
        bert_model = None

    Log_file.info('build model...')
    model = Model(args, item_num,  use_modal, bert_model).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)


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

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    run_eval(model, item_content, users_history_for_test, users_test, 512, item_num, use_modal,
                         args.mode, is_early_stop, local_rank)


def run_eval(model, item_content, user_history, users_eval, batch_size, item_num, use_modal,
             mode, is_early_stop, local_rank):
    Log_file.info('Validating...')
    if use_modal:
        item_embeddings = get_bert_embeddings(model, item_content, batch_size, args, local_rank)
    else:
        item_embeddings = get_id_embeddings(model, item_content, batch_size, args, local_rank)
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
        model_load = args.bert_model_load
        dir_label = args.behaviors + f' _{model_load}_freeze_{args.freeze_paras_before}'
        log_paras = f'{args.item_tower}-{model_load}-{args.freeze_paras_before}' \
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
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if 'train' in args.mode:
        train(args, is_use_modal, local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
