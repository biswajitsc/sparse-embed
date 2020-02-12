export CUDA_VISIBLE_DEVICES="1"
python train.py \
--model_dir=checkpoints/kubernetes/cifar/dense_64_run_1 \
--learning_rate=1e-4 --optimizer=adam \
--data_dir=../cifar-100 --evaluate=True --num_classes=101 \
--model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=none \
--max_steps=20000 --decay_step=10000

# export CUDA_VISIBLE_DEVICES="1"
# python train.py \
# --model_dir=checkpoints/kubernetes/cifar/dense_64_run_2 \
# --learning_rate=1e-4 --optimizer=adam \
# --data_dir=../cifar-100 --evaluate=True --num_classes=101 \
# --model=cifar100 --batch_size=512 --embedding_size=64 --final_activation=none \
# --max_steps=20000 --decay_step=10000
