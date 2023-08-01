import os

root_data_dir = '../../'
dataset = 'dataset/HM'
behaviors = 'hm_50w_users.tsv'
images = 'hm_50w_items.tsv'
lmdb_data = 'hm_50w_items.lmdb'
logging_num = 4
testing_num = 1

CV_resize = 224
CV_model_load = 'swin_tiny'

mode = 'test'
item_tower = 'modal'

drop_rate_list = [0.1]
embedding_dim_list = [512]
lr_list = [1e-4]
l2_weight_list = [0.1]
fine_tune_lr_list = [0]

freeze_paras_before_list = [0]
batch_size_list = [64]

dnn_layers_list = [8]


load_ckpt_name = 'epoch-92.pt'

for l2_weight in l2_weight_list:
    for batch_size in batch_size_list:
        for drop_rate in drop_rate_list:
            for lr in lr_list:
                for embedding_dim in embedding_dim_list:
                    for fine_tune_lr in fine_tune_lr_list:
                        for dnn_layer in dnn_layers_list:
                            label_screen = '{}_modality-{}_bs-{}_ed-{}_lr-{}_dp-{}_L2-{}_Flr-{}_ckp-{}'.format(
                                item_tower, CV_model_load, batch_size, embedding_dim, lr,
                                drop_rate, l2_weight, fine_tune_lr, load_ckpt_name)
                            run_test_py = "CUDA_VISIBLE_DEVICES='0' \
                                      /opt/anaconda3/bin/python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1236\
                                      run_test.py --root_data_dir {}  --dataset {} --behaviors {} --images {}  --lmdb_data {}\
                                      --mode {} --item_tower {} --load_ckpt_name {} --label_screen {}\
                                      --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} \
                                      --CV_resize {} --CV_model_load {} --dnn_layer {} --fine_tune_lr {}".format(
                                root_data_dir, dataset, behaviors, images, lmdb_data,
                                mode, item_tower, load_ckpt_name, label_screen,
                                l2_weight, drop_rate, batch_size, lr, embedding_dim,
                                CV_resize, CV_model_load, dnn_layer, fine_tune_lr)
                            os.system(run_test_py)
