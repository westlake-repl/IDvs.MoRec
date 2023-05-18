import numpy as np
import torch


def read_behaviors(behaviors_path, before_item_id_to_keys, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file):
    Log_file.info("##### images number {} {} (before clearing)#####".
                  format(len(before_item_id_to_keys), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    before_seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            before_seq_num += 1
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))
    Log_file.info("##### user seqs before {}".format(before_seq_num))

    item_id = 1
    item_id_to_keys = {}
    item_id_before_to_now = {}
    item_name_to_id = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_keys[item_id] = before_item_id_to_keys[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {}, {}#####".format(item_num, item_id - 1, len(item_id_to_keys), len(item_id_before_to_now)))
    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2]
        valid = user_seq[-(max_seq_len+2):-1]
        test = user_seq[-(max_seq_len+1):]

        users_train[user_id] = train
        users_valid[user_id] = valid
        users_test[user_id] = test

        users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    return item_num, item_id_to_keys, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_images(images_path):
    item_id_to_keys = {}
    item_name_to_id = {}
    item_id_to_name = {}
    index = 1
    with open(images_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            image_name = splited[0]
            item_name_to_id[image_name] = index
            item_id_to_name[index] = image_name
            item_id_to_keys[index] = u'{}'.format(int(image_name.replace('v', ''))).encode('ascii')
            index += 1
    return item_id_to_keys, item_name_to_id, item_id_to_name
