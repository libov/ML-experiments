# General
- Understand GAN again
- VAE - is norm and loss calculation correct? check against overleaf
- test random crops - implemented correctly for MNIST?
- need random crops switch for cifar, like for MNIST?
- ResNetBase: check batch/skip/relu ordering
- Group Norm vs Batch Norm vs Layer Norm vs .... -- and how it is related to input image normalization
- Should validation set contain crops or be more like test set? --> DONE for MNIST
- Adam vs AdamW vs SGD /// weight decay role
- drop_last = True? train vs test/val (False for the latter). What about loss calculation - any potential errors? what about VAE loss calculation? (I divide by FIRST batch size....?)
- check whetehr upsampling and skip connections there are ok

# Classification:
- try longer training (200-300 epochs) -- DONE
- try SGD, AdamW -- DONE
- try more augmentation
- try the above to see if cifar10 can be pushed closer to 95% (best so far is ~92% after 50 epochs) -- DONE
   - Note: with SGD and weight decay, it is now ~94% on the test set
- check parameter count. Autoencoder MNIST = 13M ?? why?
- try a different data set (ImageNet? CelebA? STL10? TinyImageNet? Basically something with more high res images)

# Generative
- AE/VAE too blurry - try adversarial or perceptual loss; image-to-image (Isola et.al)? SRGAN.
   - or VQ-VAE (Vector Quantized VAE) (Oord 2017)

# Technical:
- train_autoencoder and train_classifier overlap a lot. Refactor! --- DONE
- but new todo: train/val loss code duplication in train_classifier.py.
   - Also: do we need "proper" train loss calculation or rather averaged? (i.e. calculate for each batch/optimizer set, not after the full epoch like now - would save some training time)
- other architecture (mobilenetv2, efficientnet, depthwise conv; try removing residual connection; add squeeze-and-excitation)
- there is a clash between parameter name "norm" : denotes both input image normalization AND layer normalization! Need to separate the two!

# Pipeline
- Diffusion Models
- Normalizing Flows
- Vision Transformer
- YOLO and other downstream tasks
- Video tasks
- Audio Tasks
- NLP Tasks
- Reinforcement Learning
- World Models
- JEPA
- Masked Autoencoders
- Azure ML or other cloud-based training?
