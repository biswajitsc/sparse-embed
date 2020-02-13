export CUDA_VISIBLE_DEVICES="0,1,2,3"

horovodrun -np 4 -H localhost:4 \
python train.py \
--model_dir=checkpoints/msceleb/sparse_resnet_flops_run_1 \
--learning_rate=1e-3 --decay_step=160000 --optimizer=mom --momentum=0.9 \
--data_dir=../msceleb1m/tfrecords/train --evaluate=True \
--num_classes=85743 --batch_size=64 --model=faceresnet \
--embedding_size=1024 --sparsity_type=l1_norm  --max_steps=230000 \
--l1_weighing_scheme=dynamic_4 --l1_parameter=3.0 -l1_p_steps=100000 \
--evaluate=True --final_activation=soft_thresh;
