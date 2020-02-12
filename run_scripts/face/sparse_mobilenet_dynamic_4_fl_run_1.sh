# CUDA_VISIBLE_DEVICES="2"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TMPDIR="/home/scratch/bparia/tmp"
mpirun -np 4 \
-H localhost:4 \
-bind-to none -map-by slot \
-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
-mca pml ob1 -mca btl ^openib \
python train.py \
--model_dir=checkpoints/kubernetes/msceleb/sparse_mobilenet_dynamic_4_fl_run_1 \
--learning_rate=1e-4 --optimizer=mom --momentum=0.9 \
--data_dir=../msceleb1m/tfrecords/train --evaluate=True \
--num_classes=85743 --batch_size=256 --model=mobilefacenet \
--embedding_size=1024 --sparsity_type=flops_sur \
--l1_weighing_scheme=dynamic_4 --l1_parameter=100.0 \
--evaluate=True --final_activation=relu6
