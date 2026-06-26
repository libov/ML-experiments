# General
- test random crops - implemented correctly for MNIST?
- need random crops switch for cifar, like for MNIST?
- ResNetBase: check batch/skip/relu ordering
- Group Norm vs Batch Norm vs Layer Norm vs ....
- Should validation set contain crops or be more like test set? --> DONE for MNIST
- Adam vs AdamW vs SGD /// weight decay role

# Classification:
- try longer training (200-300 epochs) -- DONE
- try SGD, AdamW -- DONE
- try more augmentation
- try the above to see if cifar10 can be pushed closer to 95% (best so far is ~92% after 50 epochs)
   - EDIT: with SGD and weight decay, it is now ~94% on the test set
- check parameter count. Autoencoder MNIST = 13M ?? why?
- try a different data set (ImageNet? CelebA? STL10? TinyImageNet? Basically something with more high res images)

# Technical:
- train_autoencoder and train_classifier overlap a lot. Refactor! --- DONE
- but new todo: train/val loss code duplication in train_classifier.py.
   - Also: do we need "proper" train loss calculation or rather averaged? (i.e. calculate for each batch/optimizer set, not after the full epoch like now - would save some training time)
- other architecture (mobilenetv2, efficientnet, depthwise conv; try removing residual connection; add squeeze-and-excitation)
