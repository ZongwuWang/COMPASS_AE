# Description: run demo for training and inference
# Author: Zongwu Wang
# Usage: bash run_demo.sh [dataset_name] [action]
# log_dir_prefix="/root/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/mylogs/"
log_dir_prefix="/root/ReRAM_SNN_Acce/mylogs/"
dataset_name="MNIST"
if [ "$1"x != "x" ]; then
	dataset_name=$1
fi
action="infer"
if [ "$2"x != "x" ]; then
	action=$2
fi
echo "dataset_name: $dataset_name"
echo "action: $action"

if [ "$dataset_name"x == "MNIST"x ]; then
	python3 ./codes/${action}_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name $dataset_name -log_dir_prefix $log_dir_prefix -T 100 -max_epoch 1024 -detach_reset
elif [ "$dataset_name"x == "CIFAR10"x ]; then
	python3 ./codes/${action}_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name $dataset_name -log_dir_prefix $log_dir_prefix -T 8 -max_epoch 1024 -detach_reset
elif [ "$dataset_name"x == "FashionMNIST"x ]; then
	python3 ./codes/${action}_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name $dataset_name -log_dir_prefix $log_dir_prefix -T 8 -max_epoch 1024 -detach_reset
elif [ "$dataset_name"x == "NMNIST"x ]; then
	python3 ./codes/${action}_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name $dataset_name -log_dir_prefix $log_dir_prefix -T 10 -max_epoch 1024 -detach_reset -channels 128 -number_layer 2 -split_by number -normalization None -use_plif
elif [ "$dataset_name"x == "CIFAR10DVS"x ]; then
	# python3 ./codes/${action}_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name $dataset_name -log_dir_prefix $log_dir_prefix -T 10 -max_epoch 1024 -detach_reset -channels 128 -number_layer 4 -split_by number -normalization None -use_plif
	CUDA_DEVICE_ORDER="PCI_BUS_ID" OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8  --nnodes=1 ./codes/${action}_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name $dataset_name -log_dir_prefix $log_dir_prefix -T 100 -max_epoch 1024 -detach_reset -channels 128 -number_layer 4 -split_by number -normalization None -use_plif -batch_size 1
elif [ "$dataset_name"x == "DVS128Gesture"x ]; then
	python3 ./codes/${action}_val.py -init_tau 2.0 -use_max_pool -device cuda:0 -dataset_name $dataset_name -log_dir_prefix $log_dir_prefix -T 20 -max_epoch 1024 -detach_reset -channels 128 -number_layer 5 -split_by number -normalization None -use_plif
else
	echo "dataset_name: $dataset_name is not supported!"
fi

# ===== MNIST =====
# python3 ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name MNIST -log_dir_prefix /root/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/mylogs/ -T 8 -max_epoch 1024 -detach_reset
# python3 ./codes/infer_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name MNIST -log_dir_prefix /root/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/mylogs/ -T 8 -max_epoch 1024 -detach_reset

# ===== CIFAR10 =====
# python3 ./codes/train_val.py -init_tau 2.0 -use_max_pool -use_plif -device cuda:0 -dataset_name CIFAR10 -log_dir_prefix /root/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/mylogs/ -T 8 -max_epoch 1024 -detach_reset

