export CUDA_VISIBLE_DEVICES="1"
set -e
for i in {3..8}
do
python get_embeddings.py --num_classes=85743 --embedding_size=1024 \
--model_dir=checkpoints/kubernetes/msceleb/sparse_mobilenet_dynamic_4_fl_run_$i \
--data_dir=../megaface_distractors/tfrecords_official/ --is_training=False \
--final_activation=relu --model=mobilefacenet --debug=False
done

for i in {1..6}
do
python get_embeddings.py --num_classes=85743 --embedding_size=1024 \
--model_dir=checkpoints/kubernetes/msceleb/sparse_mobilenet_dynamic_4_l1_run_$i \
--data_dir=../megaface_distractors/tfrecords_official/ --is_training=False \
--final_activation=relu --model=mobilefacenet --debug=False
done
