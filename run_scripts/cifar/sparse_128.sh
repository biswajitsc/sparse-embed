# export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TMPDIR="/home/scratch/bparia/tmp"
mpirun -np 4 \
-H localhost:4 \
-bind-to none -map-by slot \
-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
-mca pml ob1 -mca btl ^openib \
python train.py \
--model_dir=checkpoints/kubernetes/cifar/sparse_128 \
--learning_rate=1e-3 --optimizer=adam \
--data_dir=../cifar-100-split --evaluate=True --num_classes=101 \
--model=cifar100 --batch_size=512 --embedding_size=128 \
# --sparsity_type=flops_sur --l1_weighing_scheme=dynamic_4 \
# --l1_parameter=1.0 --l1_p_steps=15e3
