set -e

for i in {1..6};
do
    python hdf5_to_bin.py ../embeddings_reranking/facescrub_sparse_mobilenet_dynamic_4_l1_run_${i}.hdf5 ../embeddings_bin
    python hdf5_to_bin.py ../embeddings_reranking/megaface_sparse_mobilenet_dynamic_4_l1_run_${i}.hdf5 ../embeddings_bin
done

for i in {3..8};
do
    python hdf5_to_bin.py ../embeddings_reranking/facescrub_sparse_mobilenet_dynamic_4_fl_run_${i}.hdf5 ../embeddings_bin
    python hdf5_to_bin.py ../embeddings_reranking/megaface_sparse_mobilenet_dynamic_4_fl_run_${i}.hdf5 ../embeddings_bin
done