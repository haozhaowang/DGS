# -*- coding: utf-8 -*-
import argparse
import math
import os
import random
import sys
import time

import torch
import torch.distributed.deprecated as dist
from cjltest.models import MnistCNN, AlexNetForCIFAR
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import test_model
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import ResNetOnCifar10
import mldatasets

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=0)
parser.add_argument('--workers-num', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='MnistCNN')

# 参数信息
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--train-bsz', type=int, default=200)
parser.add_argument('--loops', type=int, default=75)
parser.add_argument('--lr', type=float, default=0.1)
args = parser.parse_args()


# noinspection PyTypeChecker
def run(rank, model, train_pics, train_bsz):
    workers = [v+1 for v in range(args.workers_num)]
    _group = [w for w in workers].append(rank)
    group = dist.new_group(_group)

    for p in model.parameters():
        scatter_p_list = [p.data for _ in range(len(workers) + 1)]
        dist.scatter(tensor=p.data, scatter_list=scatter_p_list, group=group)

    print('Model Sent Finished!')

    ratio_list = [torch.tensor([0.0]) for _ in range(len(workers) + 1)]
    dist.gather(tensor=torch.tensor([0.0]), gather_list=ratio_list, group=group)
    ratio = sum(ratio_list)/len(workers)

    print('Begin!')

    epoch_train_loss = 0.0
    trainloss_file = './trainloss' + args.model + '.txt'
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    f_trainloss = open(trainloss_file, 'a')

    tmp = [(0, 0) for _ in range(int(math.ceil(train_pics / (len(workers) * train_bsz))))]
    s_time = time.time()
    epoch_time = s_time
    global_clock = 0
    for epoch in range(args.epochs):
        for batch_idx, (_, _) in enumerate(tmp):
            # receive the list of train loss from workers
            info_list = [torch.tensor([0.0]) for _ in range(len(workers) + 1)]
            dist.gather(tensor=torch.tensor([0.0]), gather_list=info_list, group=group)
            batch_loss = sum(info_list).item()/len(workers)
            epoch_train_loss +=  batch_loss

            param_idx = 0
            for layer_idx, param in enumerate(model.parameters()):
                tensor = torch.zeros_like(param.data)
                # FIXME FIXED：gather_list中的每个Tensor都必须是新的对象，否则会出问题
                gather_list = [torch.zeros_like(param.data) for _ in range(len(workers) + 1)]
                dist.gather(tensor=tensor, gather_list=gather_list, group=group)
                g_avg = sum(gather_list) / len(workers)
                param.data -= g_avg
                scatter_list = [param.data for _ in range(len(workers) + 1)]
                dist.scatter(tensor=g_avg, scatter_list=scatter_list, group=group)
                param_idx += 1
            global_clock += 1

            f_trainloss.write(str(args.this_rank) +
                              "\t" + str(epoch_train_loss / float(batch_idx + 1)) +
                              "\t" + str(batch_loss) +
                              "\t" + str(0) +
                              "\t" + str(0) +
                              "\t" + str(epoch) +
                              "\t" + str(0) +
                              "\t" + str(0) +
                              "\t" + str(0) +
                              "\t" + str(ratio.item()) +
                              "\t" + str(ratio.item()) +
                              "\t" + str(global_clock) +
                              '\n')

            #print('Done {}/{}!'.format(batch_idx, len(tmp)))
        print('Done Epoch {}/{}!'.format(epoch + 1, args.epochs))

        e_epoch_time = time.time()

        # test_acc, batch_interval, batch_comp_interval, batch_comm_interval
        logs = torch.tensor([0.0, 0.0, 0.0, 0.0])
        logs_list = [torch.zeros_like(logs) for _ in range(len(workers)+1)]
        #dist.gather(tensor = logs, gather_list = logs_list, group = group)
        test_acc, batch_interval, batch_comp_interval, batch_comm_interval = zip(*logs_list)
        test_acc = sum(test_acc)/len(workers)
        batch_interval = sum(batch_interval)/len(workers)
        batch_comp_interval = sum(batch_comp_interval)/len(workers)
        batch_comm_interval = sum(batch_comm_interval)/len(workers)

        #f_trainloss.write(str(args.this_rank) +
         #                 "\t" + str(epoch_train_loss / float(batch_idx+1)) +
          #                "\t" + str(0) +
           #               "\t" + str(e_epoch_time - epoch_time) +
            #              "\t" + str(e_epoch_time - s_time) +
             #             "\t" + str(epoch) +
              #            "\t" + str(test_acc.item()) +
               #           "\t" + str(batch_interval.item()) +
                #          "\t" + str(batch_comp_interval.item()) +
                 #         "\t" + str(batch_comm_interval.item()) +
                  #        "\t" + str(ratio.item()) +
                   #       "\t" + str(global_clock) +
                    #      '\n')
        f_trainloss.flush()
        epoch_time = e_epoch_time
        epoch_train_loss = 0.0

        if (epoch + 1) % 2 == 0:
            if not os.path.exists('model_state'):
                os.makedirs('model_state')
            torch.save(model.state_dict(),
                       'model_state' + '/' + args.model + '_' + str(epoch + 1) + '.pkl')

    f_trainloss.close()


def init_processes(rank, size,
                   model, train_pics, train_bsz,
                   fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, model, train_pics, train_bsz)


if __name__ == '__main__':
    # 随机数设置
    manual_seed = 1
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    workers = [v+1 for v in range(args.workers_num)]

    if args.model == 'MnistCNN':
        model = MnistCNN()
        train_pics = 60000
    elif args.model == 'LROnMnist':
        model = ResNetOnCifar10.LROnMnist()
        train_transform, test_transform = get_data_transform('mnist')
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=test_transform)
        train_pics = 60000
    elif args.model == 'AlexNet':
        if args.data_name == 'cifar10':
            model = AlexNetForCIFAR()
        else:
            model = AlexNetForCIFAR(num_classes=100)
        train_pics = 50000
    elif args.model == 'LROnCifar10':
        model = ResNetOnCifar10.LROnCifar10()
        train_transform, test_transform = get_data_transform('cifar')
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                      transform=test_transform)
        train_pics = 50000
    elif args.model == 'LROnCifar100':
        model = ResNetOnCifar10.LROnCifar100()
        train_transform, test_transform = get_data_transform('cifar')
        test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=True,
                                      transform=test_transform)
        train_pics = 50000
    elif args.model == 'ResNet18OnCifar10':
        model = ResNetOnCifar10.ResNet18()

        train_transform, test_transform = get_data_transform('cifar')
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                        transform=test_transform)
        train_pics = 50000
    elif args.model == 'ResNet34':
        model = models.resnet34(pretrained=False)

        test_transform, train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_dataset = datasets.ImageFolder(args.data_dir, train=False, download=False,
                                        transform=test_transform)
        train_pics = 121187
    elif args.model == 'Abalone':
        model = ResNetOnCifar10.abalone_model()
        test_dataset = mldatasets.abalone(args.data_dir, False)
        train_pics = 3342
    elif args.model == "Bodyfat":
        model = ResNetOnCifar10.bodyfat_model()
        test_dataset = mldatasets.bodyfat(args.data_dir, False)
        train_pics = 202
    elif args.model == 'Housing':
        model = ResNetOnCifar10.housing_model()
        test_dataset = mldatasets.housing(args.data_dir, False)
        train_pics = 405
    else:
        print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
        sys.exit(-1)

    train_bsz = args.train_bsz
    train_bsz /= len(workers)

    world_size = args.workers_num + 1
    this_rank = args.this_rank

    p = TorchProcess(target=init_processes, args=(this_rank, world_size,
                                                  model, train_pics, train_bsz,
                                                  run))
    p.start()
    p.join()
