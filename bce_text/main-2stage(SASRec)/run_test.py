import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, \
    RobertaTokenizer, RobertaModel, RobertaConfig, \
    DebertaTokenizer, DebertaModel, DebertaConfig

from parameters import parse_args
from model import Model, Bert_Encoder
from data_utils import read_news, read_news_bert, get_doc_input_bert, \
    read_behaviors, BuildTrainDataset, eval_model, get_item_embeddings, get_item_word_embs
from data_utils.utils import *
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test(args, local_rank):
    if 'roberta' in args.bert_model_load:
        Log_file.info('load roberta model...')
        bert_model_load = '/yuanzheng/id_modal/pretrained_models/roberta/' + args.bert_model_load
        tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
        config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
        bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
        if 'base' in args.bert_model_load:
            args.word_embedding_dim = 768
    else:
        Log_file.info('load bert model...')
        bert_model_load = '/yuanzheng/id_modal/pretrained_models/bert/' + args.bert_model_load
        tokenizer = BertTokenizer.from_pretrained(bert_model_load)
        config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
        bert_model = BertModel.from_pretrained(bert_model_load, config=config)

        if 'tiny' in args.bert_model_load:
            args.word_embedding_dim = 128
        if 'mini' in args.bert_model_load:
            args.word_embedding_dim = 256
        if 'small' in args.bert_model_load:
            args.word_embedding_dim = 512
        if 'medium' in args.bert_model_load:
            args.word_embedding_dim = 512
        if 'base' in args.bert_model_load:
            args.word_embedding_dim = 768
        if 'large' in args.bert_model_load:
            args.word_embedding_dim = 1024

    for index, (name, param) in enumerate(bert_model.named_parameters()):
        param.requires_grad = False
    Log_file.info('read news...')
    before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_news_bert(
        os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)

    Log_file.info('read behaviors...')
    item_num, item_id_to_dic, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, item_name_to_id, user_name_to_id = \
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

    Log_file.info('Bert Encoder...')
    bert_encoder = Bert_Encoder(args=args, bert_model=bert_model).to(local_rank)

    Log_file.info('get bert output...')
    item_word_embs = get_item_word_embs(bert_encoder, item_content, 512, args, local_rank)

    Log_file.info('build model...')
    model = Model(args, item_num).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    Log_file.info('load ckpt if not None...')
    ckpt_path = get_checkpoint(model_dir, args.load_ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    Log_file.info('load checkpoint...')
    model.load_state_dict(checkpoint['model_state_dict'])
    Log_file.info(f"Model loaded from {ckpt_path}")
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    is_early_stop = False

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    Log_file.info('\n')
    Log_file.info('Testing...')

    run_eval_test(model, item_word_embs, users_history_for_test, users_test, 512, item_num,
                 args.mode, local_rank)


def run_eval_test(model, item_word_embs, user_history, users_eval, batch_size, item_num,
             mode, local_rank):
    eval_start_time = time.time()
    Log_file.info('Validating...')
    item_embeddings = get_item_embeddings(model, item_word_embs, batch_size, args, local_rank)
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

    model_load = args.bert_model_load
    dir_label = str(args.item_tower) + f'_{model_load}'
    if args.dnn_layer == 0:
        log_paras = f'{args.item_tower}_{model_load}_bs_{args.batch_size}'\
                    f'_ed_{args.embedding_dim}_lr_{args.lr}'\
                    f'_L2_{args.l2_weight}_dp_{args.drop_rate}_Flr_{args.fine_tune_lr}'
    else:
        log_paras = f'{args.item_tower}_{model_load}_dnn_{args.dnn_layer}_bs_{args.batch_size}'\
                    f'_ed_{args.embedding_dim}_lr_{args.lr}'\
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
