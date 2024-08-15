from matplotlib.pyplot import cla
import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data.distributed import DistributedSampler
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets import split_to_train_test_set
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime
from myexamples.hooks import *

def main():
    # python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8
    # python3 -m spikingjelly.activation_based.examples.classify_dvsg_ddp -T 100 -device cuda:1 -b 1 -epochs 64 -dataset_name CIFAR10DVS -data-dir /root/ReRAM_SNN_Acce/codes/datasets/CIFAR10DVS/ -log-dir-prefix mylogs -opt adam -lr 0.001
    # CUDA_DEVICE_ORDER="PCI_BUS_ID" OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc-per-node=8 -- -m spikingjelly.activation_based.examples.classify_dvsg_ddp -T 300 -device cuda:0 -b 16 -epochs 64 -dataset_name DVS128Gesture -data-dir /root/ReRAM_SNN_Acce/codes/datasets/DVS128Gesture/ -log-dir-prefix mylogs -opt adam -lr 0.001

    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-dataset_name', type=str)
    parser.add_argument('-data-dir', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-log-dir-prefix', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    # parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-mode', default='infer', type=str, help="train or infer")

    args = parser.parse_args()

    if "RANK" in os.environ:
        ddp = True
        local_rank = int(os.environ["LOCAL_RANK"])
        # 每个进程根据自己的local_rank设置应该使用的GPU
        torch.cuda.set_device(local_rank)
        local_device = torch.device('cuda', local_rank)
        # 初始化分布式环境，主要用来帮助进程间通信
        torch.distributed.init_process_group(backend='nccl')
        # 固定随机种子
        seed = 42
        # random.seed(seed)
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        ddp = False
        local_rank = 0
        # 每个进程根据自己的local_rank设置应该使用的GPU
        torch.cuda.set_device(local_rank)
        local_device = args.device
    
    if local_rank == 0:
        print(args)

    dir_name = f'val_{0.1}_{args.dataset_name}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}'

    log_dir = os.path.join(args.log_dir_prefix, dir_name)           # log path
    pt_dir = os.path.join(args.log_dir_prefix, 'pt_' + dir_name)     # checkpoint path
    if local_rank == 0:
        print(f"Log dir: {log_dir}")
        print(f"Checkpoint dir: {pt_dir}")

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(pt_dir):
        os.mkdir(pt_dir)


    if args.dataset_name == "DVS128Gesture":
        classes = 11
        net = parametric_lif_net.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    elif args.dataset_name == "DVS128GestureSimplify":
        classes = 11
        net = parametric_lif_net.DVSGestureNetSimplify(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    elif args.dataset_name == "CIFAR10DVS":
        classes = 10
        if args.mode == "infer":
            net = parametric_lif_net.CIFAR10DVSNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
        else:
            net = parametric_lif_net.CIFAR10DVSNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True, store_v_seq=True)

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    if local_rank == 0:
        print(net)


    net.to(local_device)

    net_without_ddp = net
    if ddp:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
        net_without_ddp = net.module

    if args.dataset_name == "DVS128Gesture" or args.dataset_name == "DVS128GestureSimplify":
        train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    elif args.dataset_name == "CIFAR10DVS":
        datasets = CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')
        if args.mode == "infer":
            # train_set, test_set = split_to_train_test_set(0.9, datasets, 10, random_split=False)
            train_set = torch.utils.data.Subset(datasets, [1201])
            test_set = torch.utils.data.Subset(datasets, [1201])
        else:
            train_set, test_set = split_to_train_test_set(0.9, datasets, 10, random_split=True)

    if ddp:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None



    if ddp:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            sampler=train_sampler,
            batch_size=args.b,
            drop_last=True,
            num_workers=args.j,
            pin_memory=True
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.b,
            shuffle=True,
            drop_last=True,
            num_workers=args.j,
            pin_memory=True
        )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )


    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net_without_ddp.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net_without_ddp.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    check_point_path = os.path.join(pt_dir, 'check_point_latest.pth')
    check_point_max_path = os.path.join(pt_dir, 'check_point_max.pth')

    # net_max_path = os.path.join(pt_dir, 'net_max.pt')
    # optimizer_max_path = os.path.join(pt_dir, 'optimizer_max.pt')
    # scheduler_max_path = os.path.join(pt_dir, 'scheduler_max.pt')
    
    if args.mode == "infer":
        assert os.path.exists(check_point_path), f"Check point {check_point_path} does not exist."
        assert args.b == 1, "Batch size must be 1 for inference."

    if os.path.exists(check_point_path):
        if local_rank == 0:
            print(f"Load checkpoint from {check_point_path}")
        checkpoint = torch.load(check_point_path, map_location=local_device)
        net_without_ddp.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

        # 遍历所有子模块
        for module in net_without_ddp.modules():
            if isinstance(module, neuron.IFNode) or isinstance(module, neuron.LIFNode):
                module.store_v_seq = True

    # out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')

    # if args.amp:
    #     log_dir += '_amp'

    # if args.cupy:
    #     log_dir += '_cupy'

    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    #     print(f'Mkdir {out_dir}.')

    if local_rank == 0:
        writer = SummaryWriter(log_dir, purge_step=start_epoch)
        with open(os.path.join(log_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
            args_txt.write('\n')
            args_txt.write(' '.join(sys.argv))

    if args.mode == "train":
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            net_without_ddp.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0
            for frame, label in train_data_loader:
                optimizer.zero_grad()
                frame = frame.to(local_device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(local_device)
                label_onehot = F.one_hot(label, classes).float()

                if scaler is not None:
                    with amp.autocast():
                        out_fr = net_without_ddp(frame).mean(0)
                        loss = F.mse_loss(out_fr, label_onehot)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out_fr = net_without_ddp(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                    loss.backward()
                    optimizer.step()

                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (out_fr.argmax(1) == label).float().sum().item()

                functional.reset_net(net_without_ddp)

            train_time = time.time()
            train_speed = train_samples / (train_time - start_time)
            train_loss /= train_samples
            train_acc /= train_samples

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()

            net_without_ddp.eval()
            test_loss = 0
            test_acc = 0
            test_samples = 0
            with torch.no_grad():
                for frame, label in test_data_loader:
                    frame = frame.to(local_device)
                    frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                    label = label.to(local_device)
                    label_onehot = F.one_hot(label, classes).float()
                    out_fr = net_without_ddp(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_fr.argmax(1) == label).float().sum().item()
                    functional.reset_net(net_without_ddp)
            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)

            save_max = False
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                save_max = True

            checkpoint = {
                'net': net_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }

            if local_rank == 0:
                if save_max:
                    torch.save(checkpoint, os.path.join(pt_dir, 'check_point_max.pth'))

                torch.save(checkpoint, os.path.join(pt_dir, 'check_point_latest.pth'))

            if local_rank == 0:
                print(args)
                print(log_dir)
                print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
                print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
                print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
    elif args.mode == "infer":
        # 注册钩子
        # 存储hook句柄的列表，以便可以移除这些hook
        handles = []
        layerAttr = layer_attr()

        # 为模型的所有子模块添加hooks
        register_hook(net_without_ddp, args.dataset_name, handles)

        net_without_ddp.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                if args.mode == "infer":
                    print(f'{label=}')
                frame = frame.to(local_device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(local_device)
                label_onehot = F.one_hot(label, classes).float()
                out_fr = net_without_ddp(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net_without_ddp)
                break
        print('-------------------------------------------------')
        # print(layer_attr)
        # 如果需要在完成后清除hooks
        for handle in handles:
            handle.remove()
        


if __name__ == '__main__':
    main()