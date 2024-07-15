# This file is based on the original training/cifar/cifar10_deepspeed.py from [microsoft/DeepSpeedExamples]
# Copyright [2023] [microsoft]
# 
# Modified by [Jeong Seong Cheol] in [2024]
# [Change the task from cifar10 to ImageNet]

# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms, models

# deepspeed
import deepspeed
from deepspeed.accelerator import get_accelerator

# other
import os
import time
import wandb
import argparse
from box import Box
from datetime import datetime
from utils.utils import seed_everything, load_config, load_json

def run(args, cmd_args):
    # set seed
    seed_everything(seed=args.seed)
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed()

    ########################################################################
    # Step1. Data Preparation.
    #
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    #
    # Note:
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    ########################################################################
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dist.get_rank() != 0:
        # Might be downloading cifar data, let rank 0 download first.
        dist.barrier()

    # Load or download cifar data.
    train_dir = args.imagenet.train
    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)

    if dist.get_rank() == 0:
        # Cifar data is downloaded, indicate other ranks can proceed.
        dist.barrier()

    ########################################################################
    # Step 2. Define the network with DeepSpeed.
    #
    # First, we define a Convolution Neural Network.
    # Then, we define the DeepSpeed configuration dictionary and use it to
    # initialize the DeepSpeed engine.
    ########################################################################
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1000)
    model_engine, optimizer , trainloader, _ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters(),
        training_data=trainset
    )

    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank
    global_rank = dist.get_rank()

    # Wandb setting
    if global_rank == 0:
        wandb.init(
            project=args.wandb.project, 
            entity=args.wandb.entity, 
            name=args.wandb.run_name, 
            config={
            "default_config": args,
            "deepspeed_config": load_json(cmd_args.deepspeed_config)
            }
        )

    # For float32, target_dtype will be None so no datatype conversion needed.
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    # Define the Classification Cross-Entropy loss function.
    criterion = nn.CrossEntropyLoss()

    ########################################################################
    #
    # Step 3. Train the network.
    #
    ########################################################################
    if global_rank == 0:
        print('Total training dataset length: ' + str(len(trainset)))
        print(f'start main loop: {args.train.epochs} epochs')

    global_step = 0
    model_engine.train() 
    
    for epoch in range(args.train.epochs):  # loop over the dataset multiple times
        running_loss = 0.0        
        for data in trainloader:
            global_step += 1
            # Get the inputs. ``data`` is a list of [inputs, labels].
            inputs, labels = data[0].to(local_device), data[1].to(local_device)
            # Try to convert to target_dtype if needed.
            if target_dtype != None:
                inputs = inputs.to(target_dtype)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()
            
            # logging per log_intarval global_step
            if global_rank == 0 and global_step % args.train.log_interval == 0:
                metrics = {'_timestamp': datetime.now().timestamp(), 
                            'train/epoch': epoch+1, 
                            'train/lr': optimizer.param_groups[0]['lr'],
                            'train/loss': running_loss / args.train.log_interval}
                wandb.log(metrics, step=global_step)
                running_loss = 0.0

            # Save checkpoint per save_interval global_step
            if global_step % args.train.save_interval == 0:
                ckpt_dir = os.path.join(args.train.save_path, args.wandb.run_name)
                os.makedirs(ckpt_dir, exist_ok=True)
                model_engine.save_checkpoint(save_dir=ckpt_dir, tag=global_step)
    
            # Exit loop when over max_global_step
            if args.train.max_global_step != -1 and global_step >= args.train.max_global_step:
                break

    if global_rank == 0:
            print("Finished Training")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()
    config = load_config(cmd_args.config)
    args = Box(config, default_box=True)
    
    run(args, cmd_args)