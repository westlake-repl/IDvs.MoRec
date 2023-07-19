from data_utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--item_tower", type=str, default="id")
    parser.add_argument("--root_data_dir", type=str, default="../",)
    parser.add_argument("--dataset", type=str, default='pinterest')
    parser.add_argument("--behaviors", type=str, default='users_log.tsv')
    parser.add_argument("--images", type=str, default='images_log.tsv')
    parser.add_argument("--lmdb_data", type=str, default='image.lmdb')

    parser.add_argument("--cold_seqs", type=str, default='None')
    parser.add_argument("--new_seqs", type=str, default='None')
    parser.add_argument("--new_items", type=str, default='None')
    parser.add_argument("--new_lmdb_data", type=str, default='None')

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--l2_weight", type=float, default=0)
    parser.add_argument("--fine_tune_l2_weight", type=float, default=0.0)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--accumulation_step", type=int, default=1)

    # ============== model parameters ==============
    parser.add_argument("--CV_model_load", type=str, default='resnet-50')
    parser.add_argument("--freeze_paras_before", type=int, default=45)
    parser.add_argument("--CV_resize", type=int, default=224)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--min_seq_len", type=int, default=5)

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--load_ckpt_name", type=str, default='None')
    parser.add_argument("--label_screen", type=str, default='None')
    parser.add_argument("--logging_num", type=int, default=8)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
