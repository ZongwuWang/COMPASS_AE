
## Artifact Appendix

### Abstract

This artifact description contains information about the complete workflow required to reproduce the evaluation results of COMPASS or evaluate your customize model and settings.
Reproducing the complete experimental results involves four steps: spiking neural network training, using pytorch hooks to generate the input trace files required by the simulator, COMPASS hardware simulation, and visualization of experimental results.
We describe in detail the complete workflow from model training to final results visualization. In addition, since the process of neural network training and trace generation is relatively slow, we provide checkpoints of different nodes of the basic model so that you can start the evaluation experiments from different nodes. Input datasets of larger-scale models for the scaling runs are not publicly available due to their size.

### Artifact check-list (meta-information)

  - Algorithm: Sparsity exploitation in spiking neural network.
  - Program: Spiking Neural Networks based on IF or LIF neuron models.
  - Backbone Model: VGGNet, ResNet
  - Data set: CIFAR10-DVS, DVS128Gesture, ImageNet
  - Hardware: Double sockets of AMD EPYC 7542 32-Core Processor with 512 GB memory, and Nvidia A100-80GB GPU (GPU only for SNN training).
  - Execution: Scripts for SNN training and architecture simulation, and Jupyter notebooks for result visualization.
  - Metrics: 
        - Area: Accelerator area for a specific SNN model.
        - Energy: Energy consumption for processing an image.
        - Latency: Latency for processing an image.
  
  - Output: detail reports for above metrics.
  - Experiments: Provide docker file for quick environment configuration and experiment evaluation.
  - How much disk space required (approximately): All data sets and trace files require approximately 200 GB of storage space.
  - How much time is needed to prepare workflow (approximately): About 30 minutes for Docker image building and dataset downloading.
  - How much time is needed to complete experiments (approximately): Training the SNN network takes more than 10 days, and architecture simulation (multi-process parallelism) takes about 1 hour, depending on the number of processes supported by the server.
  - Publicly available: Yes.
  - Code licenses (if publicly available): MIT license.

## Description

### How to access

Our source code, scripts are available on Github:
```shell
git clone https://github.com/ZongwuWang/COMPASS_AE
```

### Hardware dependencies

We evaluated the simulator on an Intel(R) Xeon(R) Gold 6258R CPU and an AMD EPYC 7542 32-core processor with 512 GB memory, and trained the SNN model on an NVIDIA A100 80GB PCIe.
We use 32 processes in parallel to evaluate our architecture simulation, which requires at least 300 GB of memory. You can adjust the parallelism of evaluation in the \textit{run\_ablation.sh} file according to the the system environment.

### Software dependencies

We evaluated COMPASS on Ubuntu 20.04 on x86-64, and the whole work based on python and C++, so the pybind11 is needed to wrap the C++ dynamic libraries. We provided docker files for whole system evaluation (\textit{Dockerfile\_gpu} which includes environments for gpu training) and architecture simulation (\textit{Dockerfile\_cpu} which only includes the environments for cpu simulation).

### Data sets

We included three data sets in our repository: CIFAR10DVS, DVS128Gesture and IMAGENET, and these three data sets can be downloaded from https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671, https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794 and http://image-net.org/, respectively.

### Models

We provided three SNN models in our work for CIFAR10DVS, DVS128Gesture and IMAGENET, respectively. And users can train these models and simulate the accelerator with the provided scripts.

## Installation

If you want to evaluate this project by training a SNN model from scratch, you need to download the data sets from the urls in \ref{sec:dataset} and set the data set paths in the \textit{setup.sh}. Specifically, you need to download the \textit{CIFAR10DVS} and \textit{DVS128Gesture} into the corresponding \textit{download} folder. Be careful that the path defined in \textit{setup.sh} is the path in docker. If you download the data sets to somewhere else, you need to mount this volume into the container.

We provide the docker files so that users can quickly install the required environments of COMPASS. The command lines are as follows:

```shell
> git clone https://github.com/ZongwuWang/COMPASS_AE
# build docker images
> cd COMPASS_AE
# launch the cpu or gpu service
> docker compose up --build -d compass_<cpu|gpu>
# enter the docker container
> docker compose exec compass_<cpu|gpu> bash
# config the data set paths
> source setup.sh
```

## Experiment workflow

Reproducing the complete experimental results involves four steps: 1) spiking neural network training, 2) using pytorch hooks to generate the input trace file required by the simulator, 3) COMPASS hardware simulation, and 4) visualization of experimental results.

### Model training and trace generation

We training the SNN models with spikingjelly framework, and it is included in \textbf{\$ReRAM\_SNN\_Acce/spikingjelly} folder. We provide script \textit{run\_demo.sh} for training and inference.

```shell
> cd COMPASS_AE/spikingjelly
# Usage: run_demo.sh <dataset> <mode> <dist>
# eg.: training for CIFAR10DVS
> source run_demo.sh CIFAR10DVS train
# eg.: CIFAR10DVS inference for generating traces
> source run_demo.sh CIFAR10DVS infer
```

After the inference process, the hook function will record the execution trace and save it to the \textbf{PyNeuroSim} folder for subsequent architecture simulation input.
Due to the long training time, we provide the trained model files and the corresponding trace files in AE for subsequent accelerator simulation. The corresponding checkpoint paths are defined in the \textit{run\_demo.sh} file.

### Hardware simulation

We extended the NeuroSim simulator for SNN simulation, and the simulator code is put in \textbf{\$ReRAM\_SNN\_Acce/PyNeuroSim} folder.
We provide a script (\textit{run\_ablation.sh}) to reproduce the results in the paper.

```shell
> cd COMPASS_AE/PyNeuroSim
> soruce run_ablation.sh
```

This will create a \textbf{logs} folder and an \textbf{err\_logs\_ablation\_[timestamp]} folder in the current directory to record simulation results. These two paths can also be overridden in the script.

### Result visualization

We provide a jupyter notebook (PyNeuroSimResultAnalysisMICRO24.ipynb) in the \textbf{\$ReRAM\_SNN\_Acce/PyNeuroSim/scripts} for result visualization. To visualize the generated results, we need to change the \textit{fdir} in the jupyter notebook to \textbf{err\_logs\_ablation\_[timestamp]} defined in \ref{sec:hwsim}.

## Evaluation and expected results

After executing the batch script in \ref{sec:hwsim}, detailed performance reports similar to NeuroSim will be generated. Run the jupyter notebook we provide to extract parameters, analyze performance, and visualize the generated report, and finally you will reproduce the Fig.11 to Fig.16 in the paper.

## Experiment customization

If you need to run a custom model and dataset, you need to build a model and train it in the spikingjelly framework. For more information on how to use the spikingjelly framework, see https://spikingjelly.readthedocs.io/zh-cn/latest/#.

We provide hooks.py in the myexamples folder to collect the trace required by the simulator.
If you need to use PyNeuroSim to evaluate the acceleration effect of a custom network, in addition to using the hook function to generate a trace, you also need to define the model structure in the PyNeuroSim folder with csv format, e.g., \textit{NetWork\_CIFAR10DVS.csv}. The format of the model structure is defined in the NetDesc class.

```python
class NetDesc():
    def __init__(self, networkFile, debug: bool=False, \
        layers: int=0, snn_only: bool = True) -> None:
        self.keys = ["IH", "IW", "Cin", "KH", "KW", \
            "Cout", "pooling", "stride", "isSNN", "T"]
        ...
```

In addition, we also provide a variety of custom parameters in the simulator. In \textbf{demo.cfg}, we configure common parameters, and users can modify them here. 