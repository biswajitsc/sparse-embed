import numpy as np
import h5py
import sys
from tqdm import tqdm

try:
    embed_fpath = sys.argv[1]
    output_dir = sys.argv[2]
    print("Reading from", embed_fpath)
except:
    print('python3 hdf5_to_bin.py [embed_fpath] [output_dir]')
    exit(0)

embedfname = embed_fpath.split('/')[-1]
idx = embedfname.rfind('.')
embedfname = embedfname[:idx]

print('Writing to', embedfname + '.bin')
#with h5py.File(embed_fpath, 'r') as fp, \
#        open(output_dir+'/'+embedfname+'.bin','wb') as fpw, open(output_dir+'/'+embedfname+'.filemap', 'w') as fpw2:

with h5py.File(embed_fpath, 'r') as fp, open(output_dir+'/'+embedfname+'.bin','wb') as fpw:
    
    N = fp['embedding'].shape[0]
    dim = fp['embedding'].shape[1]
    batch_size = 1000
    
    #Number of instances
    #print( 'total number of instances={}'.format(N) )
    fpw.write( np.int32(N) )
    
    for i in tqdm(range(N)): #number of batches
        embedding = fp['embedding'][i]
        label = fp['label'][i]

        nnz_ind = np.where(embedding > 1e-5)[0]
        nnz_vals = embedding[nnz_ind]
        
        # ind_batch = fp[KEY_IND][b:b+batch_size]
        # val_batch = fp[KEY_VAL][b:b+batch_size]
        # lab_batch = fp[KEY_LABEL][b:b+batch_size]
        #txt_batch = fp[KEY_TEXT][b:b+batch_size]
        #fname_batch = fp[KEY_FNAME][b:b+batch_size]

        #print('processing {}...'.format(b))    
        fpw.write(np.int32(label)) #label
        #fpw.write(np.int32(-1)) #label
        nnz = np.int32(len(nnz_ind))
        fpw.write(nnz) #number of nonzero elements in the instance
        fpw.write( np.int32(nnz_ind).tobytes() ) #indices
        fpw.write( np.float32(nnz_vals).tobytes() ) #values

