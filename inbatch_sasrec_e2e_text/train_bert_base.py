import os

root_data_dir = '../../'

dataset = 'Dataset/MIND-large'
behaviors = 'mind_60w_users.tsv'
news = 'mind_60w_items.tsv'
logging_num = 4
testing_num = 1

bert_model_load = 'bert_base_uncased'
freeze_paras_before = 0
news_attributes = 'title'

mode = 'train'
item_tower = 'modal'

epoch = 300
load_ckpt_name = 'None'


l2_weight_list = [0.01]
fine_tune_l2_weight_list = [0.01]
drop_rate_list = [0.1]
batch_size_list = [128]
lr_list = [1e-4]
embedding_dim_list = [512]
fine_tune_lr_list = [5e-5]

for l2_weight in l2_weight_list:
    for fine_tune_l2_weight in fine_tune_l2_weight_list:
        for batch_size in batch_size_list:
            for lr in lr_list:
                for embedding_dim in embedding_dim_list:
                    for drop_rate in drop_rate_list:
                        for fine_tune_lr in fine_tune_lr_list:
                            label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_Flr{}'.format(
                                item_tower, batch_size, embedding_dim, lr,
                                drop_rate, l2_weight, fine_tune_lr)
                            run_py = "CUDA_VISIBLE_DEVICES='0,1' \
                                     /opt/anaconda3/bin/python  -m torch.distributed.launch --nproc_per_node 2 --master_port 1234\
                                     run.py --root_data_dir {}  --dataset {} --behaviors {} --news {}\
                                     --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                     --l2_weight {} --fine_tune_l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} \
                                     --news_attributes {} --bert_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {}".format(
                                root_data_dir, dataset, behaviors, news,
                                mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                l2_weight, fine_tune_l2_weight, drop_rate, batch_size, lr, embedding_dim,
                                news_attributes, bert_model_load, epoch, freeze_paras_before, fine_tune_lr)
                            os.system(run_py)
