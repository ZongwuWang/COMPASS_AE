import torch.nn as nn
import models
import os
import numpy as np
import shutil
import torch

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
        breakpoint()
        np.savetxt(fa, np.expand_dims(output_matrix[0, :], axis=0), delimiter=",",fmt='%s')
        fa.write('\n')
    
def conv_hook_fn(layer_name):
    def conv_hook(module, input, output):
        global dataset, wl_weights, wl_activations
        print(f"dataset: {dataset}")
        print(f"layer_name: {layer_name}")
        filePrefix = "PyNeuroSim"
        input_fname = './layer_record_'+str(dataset)+'/input.'+layer_name+'.csv'
        weight_fname = './layer_record_'+str(dataset)+'/weight.'+layer_name+'.csv'
        f = open('./PyNeuroSim/layer_record_'+str(dataset)+'/trace_command.sh', "a")
        fwrite = f.write(weight_fname + ' ' + input_fname+' ')
        f.close()
        if not os.path.exists(os.path.join(filePrefix, weight_fname)):
            write_matrix_weight(module.weight.cpu().data.numpy(), os.path.join(filePrefix, weight_fname))
        if 'linear' not in layer_name:
            k = module.weight.shape[-1]
            padding = module.padding
            stride = module.stride
            write_matrix_activation_conv(stretch_input(input[0].cpu().data.numpy(),k,padding,stride), None, wl_activations, os.path.join(filePrefix, input_fname))
        else:
            write_matrix_activation_fc(input[0].cpu().data.numpy(), None, wl_activations, os.path.join(filePrefix, input_fname))
    return conv_hook

def plif_hook_fn(layer_name):
    def plif_hook(module, input, output):
        global dataset
        print(f"dataset: {dataset}")
        print(f"layer_name: {layer_name}")
        filePrefix = "PyNeuroSim"
        output_fname = './layer_record_'+str(dataset)+'/output.'+layer_name+'.csv'
        f = open('./PyNeuroSim/layer_record_'+str(dataset)+'/trace_command.sh', "a")
        fwrite = f.write(output_fname+' ')
        f.close()
        # TODO:
        if 'fc' in layer_name:
            write_matrix_output_fc(output.cpu().data.numpy(), None, 1, os.path.join(filePrefix, output_fname))
        else:
            write_matrix_output_conv(output.cpu().data.numpy(), None, 1, os.path.join(filePrefix, output_fname))
    return plif_hook

def register_hook(net, dataset_name, wl_weight=8, wl_activation=8):
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
    f.write('python NeuroSim.py NetWork_'+str(dataset)+'.csv '+str(wl_weight)+' '+str(wl_activation)+' ')
    f.close()
    net.hooked = True
    hook_handles = []
    valid_layer_num = 0
    layer_attributes = []
    if dataset_name == "MNIST" or dataset_name == "CIFAR10" or dataset_name == "NMNIST" or dataset_name == "CIFAR10DVS" or dataset_name == "DVS128Gesture":
        # for idx, layer in enumerate(net.conv):
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d):
                layer_attributes.append('conv')
                full_name = f'L{valid_layer_num}.conv.conv2d'
                handle = layer.register_forward_hook(conv_hook_fn(full_name))
                hook_handles.append(handle)
                valid_layer_num += 1
            elif isinstance(layer, models.PLIFNode):
                # full_name = f'conv.plif.{valid_layer_num}'
                if layer_attributes[-1] == 'conv':
                    full_name = f'L{valid_layer_num-1}.conv.plif'
                    layer_attributes.append('plif')
                elif layer_attributes[-1] == 'fc':
                    full_name = f'L{valid_layer_num-1}.fc.plif'
                    layer_attributes.append('plif')
                else:
                    raise NotImplementedError
                handle = layer.register_forward_hook(plif_hook_fn(full_name))
                hook_handles.append(handle)
                # valid_layer_num += 1
        # for idx, layer in enumerate(net.fc):
            elif isinstance(layer, nn.Linear):
                full_name = f'L{valid_layer_num}.fc.linear'
                layer_attributes.append('fc')
                handle = layer.register_forward_hook(conv_hook_fn(full_name))
                hook_handles.append(handle)
                valid_layer_num += 1
            # elif isinstance(layer, models.PLIFNode):
            #     full_name = f'fc.plif.{valid_layer_num}'
            #     handle = layer.register_forward_hook(plif_hook_fn(full_name))
            #     hook_handles.append(handle)
            #     valid_layer_num += 1
    return hook_handles
    

def unregister_hook(net, hook_handles):
    global dataset
    net.hooked = False
    for handle in hook_handles:
        handle.remove()
    # save net structure to file
    with open('./PyNeuroSim/layer_record_'+str(dataset)+'/NetWork_'+str(dataset)+'.log', 'w') as f:
        f.write(str(net))
