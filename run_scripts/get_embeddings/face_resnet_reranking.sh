export CUDA_VISIBLE_DEVICES="2"
set -e

python get_embeddings_reranking.py --num_classes=85743 --embedding_size=512 \
--model_dir=checkpoints/kubernetes/msceleb/dense_resnet_512_run_1 \
--data_dir=../facescrub/tfrecords_official/ --is_training=False \
--final_activation=none --model=faceresnet --debug=False  \
--filelist=/home/ubuntu/megaface_distractors/facescrub_features_list.json \
--prefix=facescrub

python get_embeddings_reranking.py --num_classes=85743 --embedding_size=512 \
--model_dir=checkpoints/kubernetes/msceleb/dense_resnet_512_run_1 \
--data_dir=../megaface_distractors/tfrecords_official/ --is_training=False \
--final_activation=none --model=faceresnet --debug=False  \
--filelist=/home/ubuntu/megaface_distractors/megaface_features_list.json_1000000_1 \
--prefix=megaface


for i in {1..4}
do
python get_embeddings_reranking.py --num_classes=85743 --embedding_size=1024 \
--model_dir=checkpoints/kubernetes/msceleb/sparse_resnet_dynamic_4_fl_run_$i \
--data_dir=../facescrub/tfrecords_official/ --is_training=False \
--final_activation=soft_thresh --model=faceresnet --debug=False  \
--filelist=/home/ubuntu/megaface_distractors/facescrub_features_list.json \
--prefix=facescrub
done

for i in {1..4}
do
python get_embeddings_reranking.py --num_classes=85743 --embedding_size=1024 \
--model_dir=checkpoints/kubernetes/msceleb/sparse_resnet_dynamic_4_l1_run_$i \
--data_dir=../facescrub/tfrecords_official/ --is_training=False \
--final_activation=soft_thresh --model=faceresnet --debug=False  \
--filelist=/home/ubuntu/megaface_distractors/facescrub_features_list.json \
--prefix=facescrub
done

for i in {1..4}
do
python get_embeddings_reranking.py --num_classes=85743 --embedding_size=1024 \
--model_dir=checkpoints/kubernetes/msceleb/sparse_resnet_dynamic_4_fl_run_$i \
--data_dir=../megaface_distractors/tfrecords_official/ --is_training=False \
--final_activation=soft_thresh --model=faceresnet --debug=False  \
--filelist=/home/ubuntu/megaface_distractors/megaface_features_list.json_1000000_1 \
--prefix=megaface
done

for i in {1..4}
do
python get_embeddings_reranking.py --num_classes=85743 --embedding_size=1024 \
--model_dir=checkpoints/kubernetes/msceleb/sparse_resnet_dynamic_4_l1_run_$i \
--data_dir=../megaface_distractors/tfrecords_official/ --is_training=False \
--final_activation=soft_thresh --model=faceresnet --debug=False  \
--filelist=/home/ubuntu/megaface_distractors/megaface_features_list.json_1000000_1 \
--prefix=megaface
done