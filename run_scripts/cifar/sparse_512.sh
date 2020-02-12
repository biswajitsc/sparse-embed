export CUDA_VISIBLE_DEVICES="0"
python train.py \
--model_dir=checkpoints/kubernetes/cifar/sparse_128 \
--learning_rate=1e-3 --optimizer=mom \
--data_dir=../cifar-100-split --evaluate=True --num_classes=101 \
--model=cifar100 --batch_size=1024 --embedding_size=128 \
--sparsity_type=flops_sur --l1_weighing_scheme=dynamic_4 \
--l1_parameter=5.0 --l1_p_steps=15e3
