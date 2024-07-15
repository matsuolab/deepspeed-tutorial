# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms, models

# other
import os
import time
import wandb
import argparse
from box import Box
from datetime import datetime
from utils.utils import seed_everything, load_config, load_json

def run(args):
    # set seed
    seed_everything(seed=args.seed)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load cifar data.
    valid_dir = args.imagenet.val
    valset = torchvision.datasets.ImageFolder(valid_dir, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.eval.batch_size, shuffle=False, num_workers=4)

    # Load model
    print(f"Loading model from {args.eval.ckpt_path}")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1000)
    state_dict = torch.load(args.eval.ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['module'])

    # Set the device
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Wandb setting
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.wandb.run_name = f"eval_{current_time}"
    wandb.init(
        project=args.wandb.project, 
        entity=args.wandb.entity, 
        name=args.wandb.run_name, 
        config=args
    )
    
    # For total accuracy.
    correct, total = 0.0, 0.0
    # For accuracy per class.
    class_correct = list(0.0 for i in range(1000))
    class_total = list(0.0 for i in range(1000))

    # Start testing.
    model.eval()
    with torch.no_grad():
        for eval_time,data in enumerate(valloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Count the total accuracy.
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            # Count the accuracy per class.
            batch_correct = (predicted == labels.to(device)).squeeze()
            for i in range(args.eval.batch_size):
                label = labels[i]
                class_correct[label] += batch_correct[i].item()
                class_total[label] += 1
            if eval_time > args.eval.max_eval:
                break

    print(f"Accuracy of the network on the {total} val images: {100 * correct / total : .0f} %")
    
    metrics = {'_timestamp': datetime.now().timestamp(), 
                'eval/acc':  100 * correct / total}
    wandb.log(metrics)

    print("Finished Evaluation")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    cmd_args = parser.parse_args()
    config = load_config(cmd_args.config)
    args = Box(config, default_box=True)
    
    run(args)