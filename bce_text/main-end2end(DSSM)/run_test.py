import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, \
    RobertaTokenizer, RobertaModel, RobertaConfig, \
    DebertaTokenizer, DebertaModel, DebertaConfig


from parameters import parse_args
from model import Model
from data_utils import read_news, read_news_bert, get_doc_input_bert, \
    read_behaviors, BuildTrainDataset, eval_model, get_user_embeddings, get_item_embeddings
from data_utils.utils import *
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test(args, use_modal, local_rank):
    if use_modal:
        if 'debertaV1' in args.bert_model_load:
            Log_file.info('load roberta model...')
            bert_model_load = '/yuanzheng/id_modal/pretrained_models/deberta/' + args.bert_model_load
            tokenizer = DebertaTokenizer.from_pretrained(bert_model_load)
            config = DebertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = DebertaModel.from_pretrained(bert_model_load, config=config)
        elif 'roberta' in args.bert_model_load:
            Log_file.info('load roberta model...')
            bert_model_load = '/yuanzheng/id_modal/pretrained_models/roberta/' + args.bert_model_load
            tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
            config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
        else:
            Log_file.info('load bert model...')
            bert_model_load = '/yuanzheng/id_modal/pretrained_models/bert/' + args.bert_model_load
            tokenizer = BertTokenizer.from_pretrained(bert_model_load)
            config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = BertModel.from_pretrained(bert_model_load, config=config)

        if 'tiny' in args.bert_model_load:
            pooler_para = [37, 38]
            args.word_embedding_dim = 128
        if 'mini' in args.bert_model_load:
            pooler_para = [69, 70]
            args.word_embedding_dim = 256
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
        item_id_to_dic, item_name_to_id = read_news_bert(
            os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)

        Log_file.info('read behaviors...')
        user_num, item_num, item_id_to_content, users_train, users_valid, train_pairs, valid_pairs, test_pairs, \
        users_history_for_valid, users_history_for_test = \
            read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors),
                           item_id_to_dic, item_name_to_id, args.min_seq_len, args.max_seq_len, Log_file)

        Log_file.info('combine news information...')
        news_title, news_title_attmask, \
        news_abstract, news_abstract_attmask, \
        news_body, news_body_attmask = get_doc_input_bert(item_id_to_content, args)

        item_content = np.concatenate([
            x for x in
            [news_title, news_title_attmask,
             news_abstract, news_abstract_attmask,
             news_body, news_body_attmask]
            if x is not None], axis=1)
    else:
        item_id_to_dic, item_name_to_id = read_news(os.path.join(args.root_data_dir, args.dataset, args.news))

        Log_file.info('read behaviors...')
        user_num, item_num, item_id_to_content, users_train, users_valid, train_pairs, valid_pairs, test_pairs, \
        users_history_for_valid, users_history_for_test \
            = read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors),
                             item_id_to_dic, item_name_to_id, args.min_seq_len, args.max_seq_len, Log_file)

        item_content = np.arange(item_num + 1)
        bert_model = None


    Log_file.info('build model...')
    model = Model(args, user_num, item_num, use_modal, bert_model).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    Log_file.info('load ckpt if not None...')
    ckpt_path = get_checkpoint(model_dir, args.load_ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    Log_file.info('load checkpoint...')
    model.load_state_dict(checkpoint['model_state_dict'])
    Log_file.info(f"Model loaded from {ckpt_path}")
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    Log_file.info('\n')
    Log_file.info('Training...')

    Log_screen.info('{} testing start'.format(args.label_screen))
    Log_file.info('')
    run_eval_test(model, item_content, users_history_for_test, test_pairs, 512, user_num, item_num,
             use_modal, args.mode, local_rank)


def run_eval_test(model, item_content, user_history, eval_pairs, batch_size,
             user_num, item_num, use_modal, mode, local_rank):
    eval_start_time = time.time()
    Log_file.info('Validating...')
    user_embeddings = get_user_embeddings(model, user_num, batch_size, args, local_rank)
    item_embeddings = get_item_embeddings(model, item_content, batch_size, args, use_modal, local_rank)
    eval_model(model, user_history, eval_pairs, user_embeddings, item_embeddings, batch_size,
                            args, item_num, Log_file, mode, local_rank)
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

    if 'modal' in args.item_tower:
        is_use_modal = True
        model_load = args.bert_model_load
        dir_label = str(args.item_tower) + f'_{model_load}_freeze_{args.freeze_paras_before}'
    else:
        is_use_modal = False
        model_load = 'id'
        dir_label = str(args.item_tower)

    log_paras = f'{model_load}_dnn_{args.dnn_layers}_bs_{args.batch_size}' \
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
        test(args, is_use_modal, local_rank)

    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
