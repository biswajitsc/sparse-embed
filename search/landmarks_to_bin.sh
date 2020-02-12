set -e

for i in 1 3 5 7;
do
    python hdf5_to_bin.py ../embeddings_reranking/landmarks_sparse_512_fl_reg_${i}0.hdf5 ../embeddings_bin
    python hdf5_to_bin.py ../embeddings_reranking/landmarks_sparse_512_l1_reg_0.${i}.hdf5 ../embeddings_bin
done