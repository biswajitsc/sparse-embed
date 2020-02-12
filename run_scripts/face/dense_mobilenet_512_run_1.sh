export CUDA_VISIBLE_DEVICES="0,1,2,3"
# until 
horovodrun -np 4 -H localhost:4 \
python train.py \
--model_dir=checkpoints/kubernetes/msceleb/dense_mobilenet_512_run_1 \
--learning_rate=1e-2 --decay_step=150000 --optimizer=mom --momentum=0.9 \
--data_dir=../msceleb1m/tfrecords/train --evaluate=True \
--num_classes=85743 --batch_size=256 --model=mobilefacenet \
--embedding_size=512 --evaluate=True --final_activation=none;
# do
#     echo "#####################################################"
#     echo ""
#     echo "Restarting ..."
#     echo ""
#     echo "#####################################################"
#     sleep 1
# done
