
import os
from PIL import Image
import numpy as np
import lmdb
import pandas as pd
import pickle
import tqdm
import torch
torch.manual_seed(123456)


class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


if __name__ == '__main__':
    print('build lmdb database')
    nc_items = pd.read_table('hm_50w_items.tsv', names=['item_id'])
    image_num = len(nc_items)
    print("all images %s" % image_num)

    lmdb_path = 'hm_50w_items.lmdb'
    isdir = os.path.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    lmdb_env = lmdb.open(lmdb_path, subdir=isdir, map_size=image_num * np.zeros((3, 224, 224)).nbytes*10,
                         readonly=False, meminit=False, map_async=True)
    txn = lmdb_env.begin(write=True)
    write_frequency = 5000

    image_file = 'hm_images'
    bad_file = {}

    lmdb_keys = []
    for index, row in tqdm.tqdm(nc_items.iterrows()):
        item_name = row[0]
        item_id = int(item_name.replace('v', ''))
        item_path = item_name + '.jpg'
        lmdb_keys.append(item_id)
        try:
            img = np.array(Image.open(os.path.join(image_file, item_path)).convert('RGB'))
            temp = LMDB_Image(img, item_id)
            txn.put(u'{}'.format(item_id).encode('ascii'), pickle.dumps(temp))
            if index % write_frequency == 0 and index != 0:
                txn.commit()
                txn = lmdb_env.begin(write=True)
        except Exception as e:
            bad_file[index] = item_id

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in lmdb_keys]
    with lmdb_env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))
    print(len(keys))
    print("Flushing database ...")
    lmdb_env.sync()
    lmdb_env.close()

    print('bad_file  ', len(bad_file))
    bad_url_df = pd.DataFrame.from_dict(bad_file, orient='index', columns=['item_id'])
    bad_url_df.to_csv('bad_file.tsv', sep='\t', header=None, index=False)
