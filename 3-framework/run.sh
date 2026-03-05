python -m src.experiments.classifier --experiment_name ResNet-CIFAR10 --dataset cifar10 --dropout 0.0 --epochs 101 --optimizer adam --learning_rate 0.001 --reduce_lr cosine_annealing --eta_min 0.0 --nruns 1
python -m src.experiments.classifier --experiment_name ResNet-CIFAR10 --dataset cifar10 --dropout 0.0 --epochs 301 --optimizer sgd  --learning_rate 0.1   --reduce_lr cosine_annealing --eta_min 0.0 --nruns 1
python -m src.experiments.classifier --experiment_name ResNet-MNIST   --dataset mnist   --dropout 0.0 --epochs 51  --optimizer adam --learning_rate 0.001 --reduce_lr cosine_annealing --eta_min 0.0 --nruns 1
#python -m src.experiments.3-autoencoder
#python -m src.analysis.test-autoencoder
