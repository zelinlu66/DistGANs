#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Andrey Prokpenko (e-mail: prokopenkoav@ornl.gov)
        : Debangshu Mukherjee (e-mail: mukherjeed@ornl.gov)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
        : Vitaliy Starchenko  (e-mail: starchenkov@ornl.gov)
        : Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it) 

Usage:
  main_GANS.py (-h | --help)
  main_GANS.py [-c CONFIG_FILE] [-m MODEL] [-e EPOCHS] [-o OPTIMIZER] [-r LEARNING_RATE] [-d DATASET] [--display] [--save] [--list]

Options:
  -h, --help                  Show this screen.
  --version                   Show version.
  --display                   Use matplotlib to plot results.
  --save                      Save the model.
  --list                      List available models.
  -c, --config=<str>          Filename containing configuration parameters
  -e, --epochs=<n>            Number of epochs [default: 100].
  -m, --model=<str>           Implementation of GANs model. Multi-layer perceptrons NN (MLP), convolutional NN (CNN), conditional MLP (C-GANs), conditional CNN (CNN-CGANs), resnet, (ResNet) [default: MLP].
  -o, --optimizer=<str>       Optimizer name [default: Jacobi].
  -r, --learning_rate=<f>     Learning rate [default: 0.01].
  -d, --dataset=<srt>         Datased used for training. MNIST, CIFAR10, CIFAR100 [default: CIFAR10]
"""
# Patch for DDP added with label-parallelism

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[...existing docstring preserved...]
"""

from docopt import docopt
import matplotlib.pyplot as plt
import sys, os
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# import mpi4py  # <-- Keep MPI for inter-label communication
import numpy as np
import subprocess
import time
import argparse

# mpi4py.rc.initialize = False
# mpi4py.rc.finalize = False
# from mpi4py import MPI

plt.rcParams.update({'font.size': 14})

from Dataloader import *

list_GANs = {}

models_dir = 'GANs_models'
models_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), models_dir)

for filename in os.listdir(models_path):
    modulename, ext = os.path.splitext(filename)
    if modulename != '__pycache__' and ext == '.py':
        subpackage = '{0}.{1}'.format(models_dir, modulename)
        obj = getattr(__import__(subpackage, globals(), locals(), [modulename]), modulename)
        list_GANs.update({obj.model_name: obj})

# MPI.Init()

def merge_args(cmdline_args, config_args):
    for key in config_args.keys():
        if key not in cmdline_args:
            sys.exit('Error: unknown key in the configuration file "{}"'.format(key))
    args = {}
    args.update(cmdline_args)
    args.update(config_args)
    return args

def get_options():
    args = docopt(__doc__, version='Competitive Gradient Descent 0.0')
    args = {key[2:]: value for key, value in args.items()}
    config_args = {}
    if args['config']:
        with open(args['config']) as f:
            config_args = yaml.load(f, Loader=yaml.FullLoader)
    for skip_option in {'config', 'help'}:
        del args[skip_option]
    return merge_args(args, config_args)

def get_mig_device_info():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "-L"], encoding='utf-8'
        )
        return result
    except Exception as e:
        return f"Error getting device info: {e}"
    
def gather_and_compute_std(value: float, device: torch.device):
    """
    Gathers `value` from all ranks and computes std (on rank 0 only).
    """
    val_tensor = torch.tensor([value], device=device)
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(val_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered, val_tensor)  # gather from all ranks
    gathered_tensor = torch.cat(gathered)  # shape: (world_size,)
    
    if dist.get_rank() == 0:
        return gathered_tensor.std().item()
    else:
        return None

if __name__ == '__main__':
    config = get_options()

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    # mpi_comm_size = MPI.COMM_WORLD.Get_size()
    # mpi_rank = MPI.COMM_WORLD.Get_rank()
    # if int(os.environ.get("RANK", 0)) == 0:
    #     import nvidia_dlprof_pytorch_nvtx

    

    if 'RANK' in os.environ:
        # original setting
        # local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # world_size = int(os.environ.get('WORLD_SIZE', 1))

        # profile setting
        local_rank = args.local_rank
        world_size = args.world_size
        mpi_rank = local_rank
        mpi_comm_size = world_size
        init_file = "/tmp/ddp_init"
        world_size = int(os.environ["WORLD_SIZE"])


        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            world_size=world_size,
            rank=int(os.environ["RANK"])
        )
        
        
        # Print for debugging
        print(f"[Rank {local_rank}] Local rank: {local_rank}")
        print(f"[Rank {local_rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[Rank {local_rank}] torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"[Rank {local_rank}] Selected device name: {torch.cuda.get_device_name(0)}")

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mpi_rank == 0:
        print('-----------------')
        print('Input parameters:')
        print(yaml.dump(config))
        print('-----------------')

    epochs = int(config['epochs'])
    optimizer_name = config['optimizer']
    learning_rate = float(config['learning_rate'])
    model_name = config['model']

    if config['dataset'] == 'MNIST':
        data = mnist_data(rand_rotation=False, max_degree=90)
        n_classes = 10
    elif config['dataset'] == 'CIFAR10':
        data = cifar10_data()
        n_classes = 10
    elif config['dataset'] == 'CIFAR100':
        data = cifar100_data()
        n_classes = 100
    elif config['dataset'] == 'IMAGENET1K':
        data = imagenet_data()
        n_classes = 1000
    else:
        raise RuntimeError('Dataset not recognized')

    # ? Filter data by label for each MPI rank (label-parallelism)
    assigned_label = mpi_rank % n_classes
    data = [sample for sample in data if sample[1] == assigned_label]

    try:
        model = list_GANs[model_name](data, n_classes, model_name, local_rank, world_size)
        model.G.to(device)
        model.D.to(device)
        # model.G = DDP(model.G, device_ids=[device.index], output_device=device.index)
        # model.D = DDP(model.D, device_ids=[device.index], output_device=device.index)
    except KeyError:
        sys.exit('\n   *** Error. Specified model name: {} is not valid.\n'.format(model_name))

    start = time.time()
    print(f"[Rank {local_rank}] Training starts: {start}")
    model.train(
        num_epochs=epochs,
        lr_x=torch.tensor([learning_rate]),
        lr_y=torch.tensor([learning_rate]),
        optimizer_name=optimizer_name,
        verbose=True,
        label_smoothing=False,
        single_number=assigned_label,
        repeat_iterations=1,
    )
    end = time.time()
    print(f"[Rank {local_rank}] Training ends: {end}")

    if mpi_rank == 0:
        if config['save']:
            print("Saving the model...")
            model.save_models()

    if config['display']:
        plt.figure()
        plt.plot(
            [x for x in range(0, len(model.D_error_real_history))],
            model.D_error_real_history,
        )
        plt.plot(
            [x for x in range(0, len(model.D_error_fake_history))],
            model.D_error_fake_history,
        )
        plt.plot(
            [x for x in range(0, len(model.G_error_history))],
            model.G_error_history,
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss function value')
        plt.legend(
            [
                'Discriminator: Loss on Real Data',
                'Discriminator: Loss on Fake Data',
                'Generator: Loss',
            ]
        )
        plt.savefig('cost_report' + str(mpi_rank) + '.png')

        averageD_error_real_history = np.zeros(len(model.D_error_real_history))
        averageD_error_fake_history = np.zeros(len(model.D_error_fake_history))
        averageG_error_history = np.zeros(len(model.G_error_history))
        stdD_error_real_history = np.zeros(len(model.D_error_real_history))
        stdD_error_fake_history = np.zeros(len(model.D_error_fake_history))
        stdG_error_history = np.zeros(len(model.G_error_history))

        for i in range(0, len(model.D_error_real_history)):
            local_val = torch.tensor(model.D_error_real_history[i], device='cuda')
            
            # Perform an all-reduce sum across all processes
            dist.all_reduce(local_val, op=dist.ReduceOp.SUM)

            # Compute average manually
            local_val /= dist.get_world_size()

            averageD_error_real_history[i] = local_val.item()

        for i in range(0, len(model.D_error_fake_history)):
            local_val = torch.tensor(model.D_error_fake_history[i], device='cuda')
            
            # Perform an all-reduce sum across all processes
            dist.all_reduce(local_val, op=dist.ReduceOp.SUM)

            # Compute average manually
            local_val /= dist.get_world_size()

            averageD_error_fake_history[i] = local_val.item()

        for i in range(0, len(model.G_error_history)):
            local_val = torch.tensor(model.G_error_history[i], device='cuda')
            
            # Perform an all-reduce sum across all processes
            dist.all_reduce(local_val, op=dist.ReduceOp.SUM)

            # Compute average manually
            local_val /= dist.get_world_size()

            averageG_error_history[i] = local_val.item()


        for i in range(len(model.D_error_real_history)):
            std_val = gather_and_compute_std(model.D_error_real_history[i], device)
            if dist.get_rank() == 0:
                stdD_error_real_history[i] = std_val
        
        for i in range(len(model.D_error_fake_history)):
            std_val = gather_and_compute_std(model.D_error_fake_history[i], device)
            if dist.get_rank() == 0:
                stdD_error_fake_history[i] = std_val
        
        for i in range(len(model.G_error_history)):
            std_val = gather_and_compute_std(model.G_error_history[i], device)
            if dist.get_rank() == 0:
                stdG_error_history[i] = std_val

        if mpi_rank == 0:
            plt.figure()
            plt.plot(
                [x for x in range(0, len(averageD_error_real_history))],
                averageD_error_real_history,
                color='b',
            )
            ci = 1.96 * stdD_error_real_history
            plt.fill_between(
                [x for x in range(0, len(averageD_error_real_history))],
                (averageD_error_real_history - ci),
                (averageD_error_real_history + ci),
                color='b',
                alpha=0.2,
            )
            plt.plot(
                [x for x in range(0, len(averageD_error_fake_history))],
                averageD_error_fake_history,
                color='orange',
            )
            ci = 1.96 * stdD_error_fake_history
            plt.fill_between(
                [x for x in range(0, len(averageD_error_fake_history))],
                (averageD_error_fake_history - ci),
                (averageD_error_fake_history + ci),
                color='orange',
                alpha=0.2,
            )
            plt.plot(
                [x for x in range(0, len(averageG_error_history))],
                averageG_error_history,
                color='g',
            )
            ci = 1.96 * stdG_error_history
            plt.fill_between(
                [x for x in range(0, len(averageG_error_history))],
                (averageG_error_history - ci),
                (averageG_error_history + ci),
                color='g',
                alpha=0.2,
            )
            plt.xlabel('Epochs')
            plt.ylabel('Loss function value')
            plt.legend(
                [
                    'Discriminator: Loss on Real Data',
                    'Discriminator: Loss on Fake Data',
                    'Generator: Loss',
                ]
            )
            plt.savefig('average_cost_report.png')

# MPI.Finalize()
