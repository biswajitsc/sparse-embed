import os
import json
import sys
import matio
import numpy as np
import h5py
from tqdm import tqdm

SUFFIX = "_sparse_1024.bin"

def main():
    data_dir = sys.argv[1]
    filelist_path = sys.argv[2]
    output_path = sys.argv[3]
    with open(filelist_path, 'r') as fp:
        filelist = json.load(fp)['path']
    
    data = h5py.File(output_path, mode='w')
    inttype = h5py.special_dtype(vlen=np.int32)
    floattype = h5py.special_dtype(vlen=np.float32)

    maxlen = len(filelist) + 1
    embedding_vals = data.create_dataset(name='embedding_vals',
        shape=(maxlen,), dtype=floattype)
    embedding_idx = data.create_dataset(name='embedding_idx',
        shape=(maxlen,), dtype=inttype)
    embedding_dense = data.create_dataset(name='embedding_dense',
        shape=(maxlen,), dtype=floattype)
    filenames = data.create_dataset(name='filenames', shape=(maxlen,), dtype='S50')
    tot_embeddings = data.create_dataset(name='tot_len', shape=(1,), dtype=np.int32)

    sparsity = []

    step = 1
    for path in tqdm(filelist):
        full_path = os.path.join(data_dir, path)
        vec = matio.load_mat(full_path + SUFFIX)

        embedding = vec.ravel()
        filename = np.string_(path)

        idxs = np.where(embedding >= 1e-8)[0]
        sparsity.append(len(idxs))

        embedding_vals[step] = [embedding[idx] for idx in idxs]
        embedding_idx[step] = [idx for idx in idxs]
        embedding_dense[step] = [emb for emb in embedding]
        filenames[step] = filename
        tot_embeddings[0] = step+1

        step += 1
    
    print("Sparsity: ", np.mean(sparsity))
    data.close()


if __name__=='__main__':
    main()
