#!/bin/bash

export CIFAR10DVS_DIR="/compass_ae/datasets/CIFAR10DVS/"
export DVS128Gesture_DIR="/compass_ae/datasets/DVS128Gesture/"
export IMAGENET_DIR="/compass_ae/datasets/imagenet/"

# build the c++ dynamic libraries
# make -C PyNeuroSim

# unzip trace files
pushd spikingjelly/PyNeuroSim/
# tar cvf - layer_record_CIFAR10DVS/ | split --bytes=2000M - layer_record_CIFAR10DVS.tar.
# tar cvf - layer_record_DVS128GestureSimplify/ | split --bytes=2000M - layer_record_DVS128GestureSimplify.tar.
# tar cvf - layer_record_IMAGENET/ | split --bytes=2000M - layer_record_IMAGENET.tar.

cat layer_record_CIFAR10DVS.tar.* | tar xvf -
cat layer_record_DVS128GestureSimplify.tar.* | tar xvf -
cat layer_record_IMAGENET.tar.* | tar xvf -

popd
