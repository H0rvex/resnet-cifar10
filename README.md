# ResNet for CIFAR-10

Simplified ResNet implementation from scratch in PyTorch, inspired by He et al. 2015 ("Deep Residual Learning for Image Recognition").

## Architecture 
- Initial conv layer (3->16 channels)
- 6 residual blocks with skip connections (16->32->64 channels)
- Stride-2 downsampling at channel transitions
- Batch normalization, adaptive average pooling
- 10-class classification head

## Techniques
- Data augmentation (random flip, random crop)
- Batch normalization
- Learning rate scheduling (StepLR)
- Weight decay (L2 regularization)

## Results
- Accuracy: 84.6% on CIFAR-10 test set
- Trained for 30 epochs with Adam optimizer

## Key Concepts
- Residual/skip connections solve vanishing gradient in deep networks
- 1x1 convolutions for channel dimension matching
- Stride-based spatial downsampling

## Further Improvements
- Longer training (50-100 epochs)
- Wider network (32->64->128 channels instead of 16->32->64)
- More residual blocks
- Cosine annealing learning rate schedule
- Advanced augmentation (Cutout, Mixup)