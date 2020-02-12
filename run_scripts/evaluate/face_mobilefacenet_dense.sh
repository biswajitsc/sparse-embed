export CUDA_VISIBLE_DEVICES="0"
python evaluate_mfacenet.py --num_classes=85743 \
--debug=False --data_dir=../msceleb1m/tfrecords/train \
--num_classes=85743 --batch_size=256 \
--model_dir=checkpoints/kubernetes/msceleb/dense_mobilenet_512_run_1 \
--model=mobilefacenet --embedding_size=512 --final_activation=none
