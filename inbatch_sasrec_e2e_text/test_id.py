import os

root_data_dir = '../../'

dataset = 'Dataset/MIND-large'
behaviors = 'mind_60w_users.tsv'
news = 'mind_60w_items.tsv'
logging_num = 4
testing_num = 1

bert_model_load = 'None'
freeze_paras_before = 0
news_attributes = 'None'

mode = 'train'
item_tower = 'id'

epoch = 50
load_ckpt_name = 'epoch-5.pt'


l2_weight = 0.1
drop_rate = 0.1
batch_size = 256
lr = 1e-4
embedding_dim = 1024

fine_tune_lr = 0
label_screen = '{}_bs{}_ed{}_lr{}_dp{}_wd{}_Flr{}'.format(
    item_tower, batch_size, embedding_dim, lr,
    drop_rate, l2_weight, fine_tune_lr)
run_py = "CUDA_VISIBLE_DEVICES='0' \
         /opt/anaconda3/bin/python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1234\
         run_test.py --root_data_dir {}  --dataset {} --behaviors {} --news {}\
         --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
         --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} \
         --news_attributes {} --bert_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {}".format(
    root_data_dir, dataset, behaviors, news,
    mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
    l2_weight, drop_rate, batch_size, lr, embedding_dim,
    news_attributes, bert_model_load, epoch, freeze_paras_before, fine_tune_lr)
os.system(run_py)
