import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np

def download_cifar10(data_path='./data'):
    # Define basic transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download training data
    trainset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    # Download test data
    testset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    
    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainset, testset, classes

def show_random_images(trainset, classes, num_images=5):
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    
    for i in range(num_images):
        # Get random image
        idx = np.random.randint(len(trainset))
        img, label = trainset[idx]
        
        # Convert tensor to numpy array and transpose to correct format
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        
        # Denormalize
        img = img * 0.5 + 0.5
        
        # Plot
        axes[i].imshow(img)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Download dataset
    trainset, testset, classes = download_cifar10()
    
    # Show some random images
    show_random_images(trainset, classes)
