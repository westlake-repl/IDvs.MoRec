import os

root_data_dir = '../../'


dataset = 'Dataset/MIND-large'
behaviors = 'mind_60w_users.tsv'
news = 'mind_60w_items.tsv'
logging_num = 16
testing_num = 4

bert_model_load = 'roberta_base'
freeze_paras_before = 0
news_attributes = 'title'


mode = 'train'
item_tower = 'modal'

epoch = 60
load_ckpt_name = 'None'

dnn_layers_list = [0]

l2_weight_list = [0.01]
dropout_list = [0.1]
batch_size_list = [512]
embedding_dim_list = [512]
lr_list = [1e-4]
fine_tune_lr_list = [5e-5, 1e-5]

for l2_weight in l2_weight_list:
    for drop_rate in dropout_list:
        for dnn_layers in dnn_layers_list:
            for embedding_dim in embedding_dim_list:
                for batch_size in batch_size_list:
                    for lr in lr_list:
                        for fine_tune_lr in fine_tune_lr_list:
                            label_screen = '{}_dnnL{}_bs{}_ed{}_lr{}_dp{}_L2{}_Flr{}'.format(
                                    item_tower, dnn_layers, batch_size, embedding_dim, lr, 
                                    drop_rate, l2_weight, fine_tune_lr)
                            run_py = "CUDA_VISIBLE_DEVICES='0' \
                                         /opt/anaconda3/bin/python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1234\
                                         run.py --root_data_dir {}  --dataset {} --behaviors {} --news {}\
                                         --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                         --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} --dnn_layers {} \
                                         --news_attributes {} --bert_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {}".format(
                                            root_data_dir, dataset, behaviors, news,
                                            mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                            l2_weight, drop_rate,batch_size, lr, embedding_dim, dnn_layers,
                                            news_attributes, bert_model_load, epoch, freeze_paras_before, fine_tune_lr)
                            os.system(run_py)
