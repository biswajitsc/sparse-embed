export CUDA_VISIBLE_DEVICES="1"
python evaluate_mfacenet.py --num_classes=85743 \
--debug=False --data_dir=../msceleb1m/tfrecords/train \
--num_classes=85743 --batch_size=256 \
--model_dir=checkpoints/kubernetes/msceleb/sparse_mobilenet_dynamic_4_l1_run_4 \
--model=mobilefacenet --embedding_size=1024 --final_activation=relu
