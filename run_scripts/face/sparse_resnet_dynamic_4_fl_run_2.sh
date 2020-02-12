export CUDA_VISIBLE_DEVICES="0,1,2,3"
until horovodrun -np 4 -H localhost:4 \
python train.py \
--model_dir=checkpoints/kubernetes/msceleb/sparse_resnet_dynamic_4_fl_run_2 \
--learning_rate=1e-3 --decay_step=160000 --optimizer=mom --momentum=0.9 \
--data_dir=../msceleb1m/tfrecords/train --evaluate=True \
--num_classes=85743 --batch_size=64 --model=faceresnet \
--embedding_size=1024 --sparsity_type=flops_sur --max_steps=230000 \
--l1_weighing_scheme=dynamic_4 --l1_parameter=50.0 -l1_p_steps=100000 \
--evaluate=True --final_activation=soft_thresh;
do
    echo "#####################################################"
    echo ""
    echo "Restarting ..."
    echo ""
    echo "#####################################################"
    sleep 1
done
