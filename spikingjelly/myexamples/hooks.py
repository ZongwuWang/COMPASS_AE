import torch.nn as nn
# import models
import os
import numpy as np
import shutil
import torch
from spikingjelly.activation_based import neuron
from myexamples.layer_attr import layer_attr
from scipy.sparse import csr_matrix

def stretch_input(input_matrix,window_size = 5,padding=(0,0),stride=(1,1)):
    # TODO: This can be simplified by torch.nn.functional.unfold
    input_shape = input_matrix.shape
    output_shape_row = int((input_shape[2] + 2*padding[0] -window_size) / stride[0] + 1)
    output_shape_col = int((input_shape[3] + 2*padding[1] -window_size) / stride[1] + 1)
    item_num = int(output_shape_row * output_shape_col)
    # output_matrix.dim = (T, N, H*W, C*K*K), the last 2 dims are transposed during writing to file, N = 1 for only the first picture in the batch, T represents the time step (File dimension: T, C*K*K, H*W)
    output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*window_size*window_size))
    iter = 0
    if (padding[0] != 0):
        input_tmp = np.zeros((input_shape[0], input_shape[1], input_shape[2] + padding[0]*2, input_shape[3] + padding[1] *2))
        input_tmp[:, :, padding[0]: -padding[0], padding[1]: -padding[1]] = input_matrix
        input_matrix = input_tmp
    for i in range(output_shape_row):
        for j in range(output_shape_col):
            for b in range(input_shape[0]):
                output_matrix[b,iter,:] = input_matrix[b, :, i*stride[0]:i*stride[0]+window_size,j*stride[1]:j*stride[1]+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1

    return output_matrix

def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

	# sign bit
    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list

def pack_bits_per_row(matrix, N):
    packed_matrix = []
    for row in matrix:
        packed_row = []
        for i in range(0, len(row), N):
            # 对于每N个位，将它们打包成一个数
            chunk = row[i:i+N]
            byte = 0
            for bit in chunk:
                byte = (byte << 1) | bit
            packed_row.append(byte)
        packed_matrix.append(packed_row)
    return np.array(packed_matrix, dtype=np.int32)

def output_compress_ratio(data):
    # data dim: T, N, Cout, <H, W>
    assert data.shape[1] == 1
    sparsity = np.sum(data == 0) / data.size
    # print(f"Bit sparsity: {sparsity * 100:.2f}%")
    # print(f"original storage: {dataOut0.size / (2 ** 23):2f} MB")
    # print(dataOut0.shape)
    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1, 1)  # T, N, Cout, H, W
    
    # T, N, Cout, H, W => (H*W)*Cout, T
    data = np.transpose(data, (3, 4, 2, 1, 0)).reshape(-1, data.shape[0]).astype(bool)

    # 按行进行pack
    data_p8 = pack_bits_per_row(data, 8)
    # print(np.sum(data_p8 == 0) / data_p8.size)
    # data_p16 = pack_bits_per_row(data, 16)
    # print(np.sum(data_p16 == 0) / data_p16.size)
    # data_p32 = pack_bits_per_row(data, 32)
    # print(np.sum(data_p32 == 0) / data_p32.size)
    # print(data_p32.shape)
    # data_p64 = pack_bits_per_row(data, 64)
    # print(np.sum(data_p64 == 0) / data_p64.size)

    data_pack = data_p8
    data_csr = csr_matrix(data_pack, dtype=np.int32)
    ori_storage = data_pack.nbytes
    csr_storage = data_csr.indices.nbytes + data_csr.indptr.nbytes / 2 + data_csr.data.nbytes / 4
    # print(f'{data_csr.indices.nbytes=}, {data_csr.indptr.nbytes=}, {data_csr.data.nbytes=}')
    # print(f"{csr_storage=}")
    # print(f"ori_storage = {ori_storage / (2 ** 20):2f} MB, csr_storage = {csr_storage / (2 ** 20):2f} MB")
    # print(f"compression ratio = {csr_storage / ori_storage * 100:.2f} %")
    return sparsity, csr_storage / ori_storage

def input_compress_ratio(data):
    # data dim: T, N, Cin, <H, W>
    # dataOut0 = LoadInSnnOutputData("../layer_record_CIFAR10DVS/output.L3.conv.plif.csv", 0, 100, 100, unmerge=True)
    sparsity = np.sum(data == 0) / data.size
    # print(f"Bit sparsity: {sparsity * 100:.2f}%")
    # print(f"original storage: {dataOut0.size / (2 ** 23):2f} MB")
    # print(dataOut0.shape)

    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1, 1)  # T, N, Cin, H, W
    
    # T, N, Cin, H, W => (H*W)*Cin, T
    data = np.transpose(data, (3, 4, 2, 1, 0)).reshape(-1, data.shape[0]).astype(bool)
    
    # T, (H*W), Cout => (H*W), Cout, T => (H*W)*Cout, T
    # data = np.transpose(data, (1, 2, 0)).reshape(-1, data.shape[0]).astype(bool)
    # 按行进行pack
    data_p8 = pack_bits_per_row(data, 8)
    # print(np.sum(data_p8 == 0) / data_p8.size)
    # data_p16 = pack_bits_per_row(data, 16)
    # print(np.sum(data_p16 == 0) / data_p16.size)
    # data_p32 = pack_bits_per_row(data, 32)
    # print(np.sum(data_p32 == 0) / data_p32.size)
    # print(data_p32.shape)
    # data_p64 = pack_bits_per_row(data, 64)
    # print(np.sum(data_p64 == 0) / data_p64.size)

    data_pack = data_p8
    data_csr = csr_matrix(data_pack, dtype=np.int32)
    ori_storage = data_pack.nbytes
    csr_storage = data_csr.indices.nbytes + data_csr.indptr.nbytes / 2 + data_csr.data.nbytes / 4
    # print(f'{data_csr.indices.nbytes=}, {data_csr.indptr.nbytes=}, {data_csr.data.nbytes=}')
    # print(f"{csr_storage=}")
    # print(f"ori_storage = {ori_storage / (2 ** 20):2f} MB, csr_storage = {csr_storage / (2 ** 20):2f} MB")
    # print(f"compression ratio = {csr_storage / ori_storage * 100:.2f} %")
    return sparsity, csr_storage / ori_storage


def write_matrix_weight(input_matrix,filename):
    cout = input_matrix.shape[0]
    # Row = C x K x K, Col = N
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    with open(filename, 'a') as fa:
        np.savetxt(fa, weight_matrix, delimiter=",",fmt='%10.5f')
        # insert a blank line
        fa.write('\n')


def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
    # only write the first picture in the batch for evaluation
    '''
    # only 0/1 exists in the input matrix in SNN, so `length` is fixed to 1
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length])
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')
    '''
    with open(filename, 'a') as fa:
        # data, scale = dec2bin(input_matrix[0, :].transpose(), length)
        # for item in data:
        #     np.savetxt(fa, item, delimiter=",",fmt='%s')
        #     fa.write('========\n')
        np.savetxt(fa, input_matrix[0, :].transpose(), delimiter=",",fmt='%s')
        fa.write('\n')

def write_matrix_output_conv(output_matrix,fill_dimension,length,filename):
    # only write the first picture in the batch for evaluation
    '''
    # only 0/1 exists in the input matrix in SNN, so `length` is fixed to 1
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length])
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')
    '''
    with open(filename, 'a') as fa:
        output_matrix = output_matrix[0, :].reshape(output_matrix.shape[1],-1)
        np.savetxt(fa, output_matrix.transpose(), delimiter=",",fmt='%s')
        fa.write('\n')

def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):
    # only write the first picture in the batch for evaluation
    '''
    # only 0/1 exists in the input matrix in SNN, so `length` is fixed to 1
    filled_matrix_b = np.zeros([input_matrix.shape[1],length])
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i] =  b
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')
    '''
    with open(filename, 'a') as fa:
        # data, scale = dec2bin(input_matrix[0, :].transpose(), length)
        # for item in data:
        #     np.savetxt(fa, item, delimiter=",",fmt='%s')
        #     fa.write('========\n')
        np.savetxt(fa, input_matrix[0, :].transpose(), delimiter=",",fmt='%s')
        fa.write('\n')

def write_matrix_output_fc(output_matrix,fill_dimension,length,filename):
    # only write the first picture in the batch for evaluation
    '''
    # only 0/1 exists in the input matrix in SNN, so `length` is fixed to 1
    filled_matrix_b = np.zeros([input_matrix.shape[1],length])
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i] =  b
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')
    '''
    with open(filename, 'a') as fa:
        np.savetxt(fa, np.expand_dims(output_matrix[0, :], axis=0), delimiter=",",fmt='%s')
        fa.write('\n')
    
def conv_hook_fn(layer_name):
    def conv_hook(module, input, output):
        global dataset, wl_weights, wl_activations
        filePrefix = "PyNeuroSim"
        layerAttr = layer_attr()
        if not layerAttr.full:
            layerAttr.layer_attr_list.append(layer_name.split('.')[0])
        layerAttr.incHeader()
        full_name = f"L{layerAttr.header + 1}." + layer_name
        print(module)
        # print(f"dataset: {dataset}")
        print(f"layer_name: {full_name}")
        input_fname = './layer_record_'+str(dataset)+'/input.'+full_name+'.csv'
        weight_fname = './layer_record_'+str(dataset)+'/weight.'+full_name+'.csv'
        # keep equal index diff for PyNeuroSim argv, there is no following plif layer after some conv or linear layers
        output_fname = './layer_record_'+str(dataset)+'/output.' + '.'.join(full_name.split('.')[:-1]) + '.plif'+'.csv'
        sparse, compress_ratio = input_compress_ratio(input[0].cpu().data.numpy())
        # print(f"sparsity: {sparse * 100:.2f}%, compress_ratio: {compress_ratio * 100:.2f}%")
        f = open('./PyNeuroSim/layer_record_'+str(dataset)+'/trace_command.sh', "a")
        # fwrite = f.write(weight_fname + ' ' + input_fname + ' ' + output_fname + ' ')
        fwrite = f.write(weight_fname + ' ' + input_fname + ' ' + '{:.3f}'.format(sparse) + ' ' + '{:.3f}'.format(compress_ratio) + ' ' + output_fname + ' ')
        f.close()
        if not os.path.exists(os.path.join(filePrefix, weight_fname)):
            write_matrix_weight(module.weight.cpu().data.numpy(), os.path.join(filePrefix, weight_fname))
        if 'linear' not in full_name:
            k = module.weight.shape[-1]
            padding = module.padding
            stride = module.stride
            for item in input[0].cpu().data.numpy():
                write_matrix_activation_conv(stretch_input(item,k,padding,stride), None, wl_activations, os.path.join(filePrefix, input_fname))
        else:
            for item in input[0].cpu().data.numpy():
                write_matrix_activation_fc(item, None, wl_activations, os.path.join(filePrefix, input_fname))
    return conv_hook

def plif_hook_fn(layer_name):
    def plif_hook(module, input, output):
        global dataset
        layerAttr = layer_attr()
        full_name = f"L{layerAttr.header + 1}.{layerAttr.layer_attr_list[layerAttr.header]}." + layer_name
        # print(f"dataset: {dataset}")
        print(f"layer_name: {full_name}")
        filePrefix = "PyNeuroSim"
        output_fname = './layer_record_' + str(dataset)+'/output.' + full_name + '.csv'
        sparse, compress_ratio = output_compress_ratio(output.cpu().data.numpy())
        print(f"sparsity: {sparse * 100:.2f}%, compress_ratio: {compress_ratio * 100:.2f}%")
        f = open('./PyNeuroSim/layer_record_'+str(dataset)+'/trace_command.sh', "a")
        # fwrite = f.write(output_fname+' ')
        fwrite = f.write("{:.3f}".format(sparse) + ' ' + "{:.3f}".format(compress_ratio) + ' ')
        f.close()
        # TODO:
        if 'fc' in full_name:
            for item in output.cpu().data.numpy():
                write_matrix_output_fc(item, None, 1, os.path.join(filePrefix, output_fname))
        else:
            for item in output.cpu().data.numpy():
                write_matrix_output_conv(item, None, 1, os.path.join(filePrefix, output_fname))
    return plif_hook

def register_hook(net, dataset_name, hook_handles, wl_weight=8, wl_activation=8):
    global dataset, wl_weights, wl_activations
    dataset= dataset_name
    wl_weights = wl_weight
    wl_activations = wl_activation
    if os.path.exists('./PyNeuroSim/layer_record_'+str(dataset)):
        shutil.rmtree('./PyNeuroSim/layer_record_'+str(dataset))
    os.makedirs('./PyNeuroSim/layer_record_'+str(dataset))
    if os.path.exists('./PyNeuroSim/layer_record_'+str(dataset)+'/trace_command.sh'):
        os.remove('./PyNeuroSim/layer_record_'+str(dataset)+'/trace_command.sh')
    f = open('./PyNeuroSim/layer_record_'+str(dataset)+'/trace_command.sh', "w")
    f.write('./NeuroSim NetWork_'+str(dataset)+'.csv '+str(wl_weight)+' '+str(wl_activation)+' ')
    f.close()
    net.hooked = True
    valid_layer_num = 0
    if dataset_name == "MNIST" or dataset_name == "CIFAR10" or dataset_name == "NMNIST" or dataset_name == "CIFAR10DVS" or dataset_name == "DVS128Gesture" or dataset_name == "IMAGENET" or dataset_name == "DVS128GestureSimplify":
        for idx, layer in enumerate(net.children()):
            if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1):     # 下采样直接忽略
                pass
            elif isinstance(layer, nn.Conv2d):
                # layer_attr.append('conv')
                # full_name = f'L{valid_layer_num}.conv.conv2d'
                full_name = "conv.conv2d"
                handle = layer.register_forward_hook(conv_hook_fn(full_name))
                hook_handles.append(handle)
                # valid_layer_num += 1
            elif isinstance(layer, neuron.IFNode) or isinstance(layer, neuron.LIFNode):
                # full_name = f'conv.plif.{valid_layer_num}'
                '''
                if layer_attr[-1] == 'conv':
                    full_name = f'L{valid_layer_num-1}.conv.plif'
                    layer_attr.append('plif')
                elif layer_attr[-1] == 'fc':
                    full_name = f'L{valid_layer_num-1}.fc.plif'
                    layer_attr.append('plif')
                else:
                    print(layer_attr)
                    raise NotImplementedError
                '''
                # full_name = f'L{valid_layer_num-1}.{layer.predecessor}.plif'
                full_name = "plif"
                handle = layer.register_forward_hook(plif_hook_fn(full_name))
                hook_handles.append(handle)
                # valid_layer_num += 1
        # for idx, layer in enumerate(net.fc):
            elif isinstance(layer, nn.Linear):
                # full_name = f'L{valid_layer_num}.fc.linear'
                full_name = "fc.linear"
                # layer_attr.append('fc')
                handle = layer.register_forward_hook(conv_hook_fn(full_name))
                hook_handles.append(handle)
                valid_layer_num += 1
            # elif isinstance(layer, models.PLIFNode):
            #     full_name = f'fc.plif.{valid_layer_num}'
            #     handle = layer.register_forward_hook(plif_hook_fn(full_name))
            #     hook_handles.append(handle)
            #     valid_layer_num += 1
            elif len(list(layer.children())) > 0:
                register_hook(layer, dataset_name, hook_handles, wl_weight, wl_activation)
    return hook_handles
    

def unregister_hook(net, hook_handles):
    global dataset
    net.hooked = False
    for handle in hook_handles:
        handle.remove()
    # save net structure to file
    with open('./PyNeuroSim/layer_record_'+str(dataset)+'/NetWork_'+str(dataset)+'.log', 'w') as f:
        f.write(str(net))
