export CUDA_VISIBLE_DEVICES="0,1,2,3"

horovodrun -np 4 -H localhost:4 \
python train.py \
--model_dir=checkpoints/msceleb/dense_resnet_512_run_1 --max_steps=180000 \
--learning_rate=1e-2 --decay_step=150000 --optimizer=mom --momentum=0.9 \
--data_dir=../msceleb1m/tfrecords/train --evaluate=True \
--num_classes=85743 --batch_size=64 --model=faceresnet \
--embedding_size=512 --evaluate=True --final_activation=none;
