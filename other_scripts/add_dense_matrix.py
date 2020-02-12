import h5py
import sys
import numpy as np
import tqdm


def main():
  filepath = sys.argv[1]
  dim = int(sys.argv[2])

  print(filepath, dim)

  maxlen = int(1.5e7)

  floattype = h5py.special_dtype(vlen=np.float32)
  data = h5py.File(filepath, mode='r+')
  embedding_dense = data.create_dataset(name='embedding_dense',
    shape=(maxlen,), dtype=floattype)

  embedding_idx = data['embedding_idx']
  embedding_vals = data['embedding_vals']

  tot_len = data['tot_len'][0]
  for i in tqdm.tqdm(range(tot_len)):
    dense = np.zeros(dim)
    idxs = embedding_idx[i]
    vals = embedding_vals[i]
    for pos, val in zip(idxs, vals):
      dense[pos] = val
    embedding_dense[i] = dense

  data.close()

if __name__=='__main__':
    main()
