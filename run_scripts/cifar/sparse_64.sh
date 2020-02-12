# soft thresh lam = 0.1 margin = 0.3 -> 0.5 l1_param decreased
# export CUDA_VISIBLE_DEVICES="1"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_1 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=25000 --sparsity_type=flops_sur --l1_parameter=0.01 --l1_p_steps=21000 \
# --decay_step=10000 --l1_weighing_scheme=dynamic_4

# soft thresh lam = 0.1 margin = 0.5
# export CUDA_VISIBLE_DEVICES="1"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_2 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=20000 --sparsity_type=flops_sur --l1_parameter=0.002 --l1_p_steps=21000 \
# --decay_step=10000 --l1_weighing_scheme=dynamic_4

# soft_thresh lam = 0.2 margin = 0.3
# export CUDA_VISIBLE_DEVICES="1"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_3 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=25000 --decay_step=10000

# soft_thresh lam = 0.1 margin = 0.3
# export CUDA_VISIBLE_DEVICES="1"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_4 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=25000 --decay_step=10000


# changing l2 regularization from 0.00004 -> 0.0001 and 0.0005 -> 0.001
# export CUDA_VISIBLE_DEVICES="3"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_5 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=25000 --decay_step=10000

# changing last layer l2 regularization from 0.001 -> 0.005
# export CUDA_VISIBLE_DEVICES="2"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_6 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=25000 --decay_step=10000

# export CUDA_VISIBLE_DEVICES="0"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_7 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=30000 --decay_step=10000 --l1_weighing_scheme=dynamic_4 \
# --l1_parameter=0.002 --l1_p_steps=12000

# ******************************
# Changing margin from 0.3 to 0.2
# export CUDA_VISIBLE_DEVICES="2"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_8 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=40000 --decay_step=10000 --l1_weighing_scheme=dynamic_4 \
# --l1_parameter=0.01 --l1_p_steps=12000

# export CUDA_VISIBLE_DEVICES="0"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_9 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=30000 --decay_step=6000 --l1_weighing_scheme=dynamic_4 \
# --l1_parameter=0.02 --l1_p_steps=20000

# export CUDA_VISIBLE_DEVICES="0"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_10 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=50000 --decay_step=10000 --l1_weighing_scheme=dynamic_4 \
# --l1_parameter=0.001 --l1_p_steps=12000

# decreasing final regularization to 0.001 and margin to 0.1
# export CUDA_VISIBLE_DEVICES="1"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_11 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=50000 --decay_step=15000 --l1_weighing_scheme=dynamic_4 \
# --l1_parameter=0.005 --l1_p_steps=15000

# export CUDA_VISIBLE_DEVICES="0"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_12 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=50000 --decay_step=15000 --l1_weighing_scheme=dynamic_4 \
# --l1_parameter=0.0 --l1_p_steps=15000

# export CUDA_VISIBLE_DEVICES="0"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/sparse_64_run_13 \
# --learning_rate=1e-3 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
# --max_steps=50000 --decay_step=15000 --l1_weighing_scheme=dynamic_4 \
# --l1_parameter=0.001 --l1_p_steps=15000 --sparsity_type=flops_sur

export CUDA_VISIBLE_DEVICES="3"
python train.py \
--model_dir=checkpoints/kubernetes/cifar/sparse_64_run_14 \
--learning_rate=1e-3 --optimizer=adam \
--data_dir=../cifar-100 --evaluate=True --num_classes=101 \
--model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=soft_thresh \
--max_steps=50000 --decay_step=15000 --l1_weighing_scheme=dynamic_4 \
--l1_parameter=0.001 --l1_p_steps=15000 --sparsity_type=l1_norm
