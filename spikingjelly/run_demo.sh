#!/bin/bash

# CIFAR10DVS_DIR="/cephfs/shared/wangzw/ReRAM_SNN_Acce/codes/datasets/CIFAR10DVS/"
# DVS128Gesture_DIR="/cephfs/shared/wangzw/ReRAM_SNN_Acce/codes/datasets/DVS128Gesture/"
# IMAGENET_DIR="/cephfs/shared/yangning/DATA_CNN/imagenet/"

# check the exist of the os environment "CIFAR10DVS_DIR", "DVS128Gesture_DIR" and "IMAGENET_DIR"
if [ -z ${CIFAR10DVS_DIR} ]; then
    # export CIFAR10DVS_DIR="/path/to/cifar10dvs"
    echo "CIFAR10DVS_DIR is not set yet, please run \$ReRAM_SNN_Acce_Root/setup.sh first!!!"
	return 1
fi
if [ -z ${DVS128Gesture_DIR} ]; then
    # export CIFAR10DVS_DIR="/path/to/cifar10dvs"
    echo "DVS128Gesture_DIR is not set yet, please run \$ReRAM_SNN_Acce_Root/setup.sh first!!!"
	return 1
fi
if [ -z ${IMAGENET_DIR} ]; then
    # export CIFAR10DVS_DIR="/path/to/cifar10dvs"
    echo "IMAGENET_DIR is not set yet, please run \$ReRAM_SNN_Acce_Root/setup.sh first!!!"
	return 1
fi

LOG_DIR="mylogs"

# new LOG_DIR if not exists
if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

# check args
if [ "$#" -lt 2 ]; then
	echo "Usage: $0 <dataset> <mode> <dist>"
	echo "dataset: CIFAR10DVS, DVS128GestureSimplify or IMAGENET"
	echo "mode: train or infer"
	echo "dist: optional, distribute training, True or False (default)"
	return 1
else
	dataset=$1
	mode=$2
	dist=${3:-False}
	echo "dataset: $dataset"
	echo "mode: $mode"
	echo "dist: $dist"
fi

if [ "$dataset" = "CIFAR10DVS" ]; then
	# check dataset
	if [ ! -d "$CIFAR10DVS_DIR" ]; then
		echo "CIFAR10DVS dataset not found, please download it from https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671"
		return 1
	fi
	python3 -m spikingjelly.activation_based.examples.classify_dvsg_ddp -T 100 -device cuda:0 -b 1 -epochs 64 -dataset_name $dataset -data-dir $CIFAR10DVS_DIR -log-dir-prefix $LOG_DIR -opt adam -lr 0.001 -mode $mode
elif [ "$dataset" = "DVS128GestureSimplify" ]; then
	# check dataset
	if [ ! -d "$DVS128Gesture_DIR" ]; then
		echo "DVS128Gesture dataset not found, please download it from https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794"
		return 1
	fi
	python3 -m spikingjelly.activation_based.examples.classify_dvsg_ddp -T 100 -device cuda:0 -b 1 -epochs 64 -dataset_name $dataset -data-dir $DVS128Gesture_DIR -log-dir-prefix $LOG_DIR -opt adam -lr 0.001 -mode $mode
elif [ "$dataset" = "IMAGENET" ]; then
	# check dataset
	if [ ! -d "$IMAGENET_DIR" ]; then
		echo "IMAGENET dataset not found, please download it from http://image-net.org/"
		return 1
	fi
	if [ "$mode" = "train" ]; then
		if [ "$dist" = "True" ]; then
			# distributed training
			OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 -- python3 -m spikingjelly.activation_based.model.train_imagenet_example --device cuda:0 --T 100 --model spiking_resnet18 --data-path $IMAGENET_DIR --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90  --output-dir $LOG_DIR
		else
			python3 -m spikingjelly.activation_based.model.train_imagenet_example --device cuda:0 --T 100 --model spiking_resnet18 --data-path $IMAGENET_DIR --batch-size 16 --lr 0.1 --lr-scheduler cosa --epochs 90  --output-dir $LOG_DIR
		fi
	else
		python3 -m spikingjelly.activation_based.model.train_imagenet_example --device cuda:0 --T 100 --model spiking_resnet18 --data-path $IMAGENET_DIR --batch-size 16 --lr 0.1 --lr-scheduler cosa --epochs 90  --output-dir $LOG_DIR --hook --resume latest --test-only	
	fi
else
	echo "dataset should be CIFAR10DVS, DVS128GestureSimplify or IMAGENET"
	return 1
fi