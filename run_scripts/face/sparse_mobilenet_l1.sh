export CUDA_VISIBLE_DEVICES="1,2,3,4"

horovodrun -np 4 -H localhost:4 \
python train.py \
--model_dir=checkpoints/msceleb/sparse_mobilenet_l1_run_1 \
--learning_rate=1e-3 --optimizer=mom --momentum=0.9 \
--l1_weighing_scheme=dynamic_4 --l1_parameter=1.0 \
--data_dir=../msceleb1m/tfrecords/train --evaluate=True \
--num_classes=85743 --batch_size=256 --model=mobilefacenet \
--embedding_size=1024 --sparsity_type=l1_norm \
--evaluate=True --final_activation=relu;
