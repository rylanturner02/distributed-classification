import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image

from model import TreeClassifier

class TreeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = self.root_dir / class_name
            for img_path in class_path.glob("*.jpg"):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def setup(rank, world_size, backend):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, data_dir, backend, batch_size=32, num_epochs=10, learning_rate=0.001):
    """
    Training function supporting both GPU and CPU
    """
    setup(rank, world_size, backend)
    
    device = torch.device('cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    dataset = TreeDataset(train_dir, transform=transform)
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=2, 
        pin_memory=False
    )
    
    model = TreeClassifier(num_classes=len(dataset.classes))
    model = model.to(device)
    model = DDP(model, device_ids=None)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict(), 'tree_classifier.pth')
                
                class_names = {i: name for i, name in enumerate(dataset.classes)}
                with open('class_names.json', 'w') as f:
                    json.dump(class_names, f)
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Distributed Tree Classifier Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # CPU distributed training
    backend = 'gloo'
    world_size = min(os.cpu_count(), 8)

    print(f"Training with {backend} backend on {world_size} CPUs")
    
    # multiprocessing to spawn processes
    torch.multiprocessing.spawn(
        train,
        args=(world_size, args.data_dir, backend, args.batch_size, args.epochs, args.lr),
        nprocs=world_size
    )

if __name__ == '__main__':
    main()
