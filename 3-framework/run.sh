python -m src.experiments.train --task classification  --experiment_name ResNet-CIFAR10         --dataset cifar10 --dropout 0.0 --epochs 101 --optimizer adam  --learning_rate 0.001 --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.0    --nruns 1
python -m src.experiments.train --task classification  --experiment_name ResNet-CIFAR10         --dataset cifar10 --dropout 0.0 --epochs 101 --optimizer adamw --learning_rate 0.001 --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.01   --nruns 1
python -m src.experiments.train --task classification  --experiment_name ResNet-CIFAR10         --dataset cifar10 --dropout 0.0 --epochs 301 --optimizer sgd   --learning_rate 0.1   --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.0    --nruns 1
python -m src.experiments.train --task classification  --experiment_name ResNet-CIFAR10         --dataset cifar10 --dropout 0.0 --epochs 301 --optimizer sgd   --learning_rate 0.1   --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.0005 --nruns 1
python -m src.experiments.train --task classification  --experiment_name ResNet-MNIST           --dataset mnist   --dropout 0.0 --epochs 51  --optimizer adam  --learning_rate 0.001 --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.0    --nruns 1

python -m src.experiments.train --task autoencoder     --experiment_name Autoencoder-MNIST      --dataset mnist   --dropout 0.0 --epochs 51  --optimizer adam  --learning_rate 0.001 --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.0    --nruns 1  --latent_dim 100
python -m src.experiments.train --task autoencoder     --experiment_name Autoencoder-CIFAR10    --dataset cifar10 --dropout 0.0 --epochs 51  --optimizer adam  --learning_rate 0.001 --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.0    --nruns 1  --latent_dim 100
python -m src.analysis.test-autoencoder --experiment_name Autoencoder-MNIST     --run_name mnist-dropout-0.0-optimizer-adam-lr-0.001-run0   --plots_dir plots/autoencoder/mnist
python -m src.analysis.test-autoencoder --experiment_name Autoencoder-CIFAR10   --run_name cifar10-dropout-0.0-optimizer-adam-lr-0.001-run0 --plots_dir plots/autoencoder/cifar10

python -m src.experiments.train --task vae  --experiment_name VAE-MNIST     --dataset mnist   --dropout 0.0 --epochs 51  --optimizer adam  --learning_rate 0.001 --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.0    --nruns 1  --latent_dim 100
python -m src.experiments.train --task vae  --experiment_name VAE-CIFAR10   --dataset cifar10 --dropout 0.0 --epochs 51  --optimizer adam  --learning_rate 0.001 --lr_scheduler cosine_annealing --eta_min 0.0 --weight_decay 0.0    --nruns 1  --latent_dim 100

python -m src.analysis.test-autoencoder --task vae --experiment_name VAE-MNIST      --run_name mnist-dropout-0.0-optimizer-adam-lr-0.001-run0   --plots_dir plots/vae/mnist
python -m src.analysis.test-autoencoder --task vae --experiment_name VAE-CIFAR10    --run_name cifar10-dropout-0.0-optimizer-adam-lr-0.001-run0 --plots_dir plots/vae/cifar10

python -m src.experiments.train --task gan  --experiment_name GAN-MNIST --dataset mnist --dropout 0.0 --epochs 51  --optimizer adam  --learning_rate 0.0001 --nruns 1  --latent_dim 100
