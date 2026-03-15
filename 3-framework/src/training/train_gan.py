import torch
import torch.nn as nn

import mlflow

import time as time

def train_gan(model, train_loader, num_epochs, lr_g=1e-4, lr_d=4e-4, n_discriminator_steps=5):

    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))
    lambda_gp = 10

    print(f"Learning rate (G): {generator_optimizer.param_groups[0]['lr']}")
    print(f"Learning rate (D): {discriminator_optimizer.param_groups[0]['lr']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    latent_dim = model.latent_dim

    start = time.time()

    for epoch in range(num_epochs):

        discriminator_loss = 0.0
        generator_loss = 0.0
        score_real = 0.0
        score_fake = 0.0
  
        for images, _ in train_loader:

            images = images.to(device)
            batch_size = images.shape[0]

            ########################################################################
            # Step 1. Train the discriminator/critic, keep the generator fixed.
            ########################################################################
            
            model.discriminator.train()
            model.generator.eval()

            # we train the discriminator(critic) for n_discriminator_steps steps, then update the generator once. 
            # This is a common training scheme for WGANs, but you can experiment with different values of n_discriminator_steps (e.g., 1, 5, 10) to see how it affects training stability and convergence.
            discriminator_optimizer.zero_grad()

            # sample a batch of latent vectors and generate fake images
            latent_vectors = torch.randn(batch_size, latent_dim, device=device)
            with torch.no_grad():
                G = model.generator(latent_vectors)

            # TODO!
            #if epoch % print_image_freq == 0 and batch == 0:
            #    plot_examples(G.view(-1, 1, 28, 28).cpu())
            
            # Sample a point along a straight line between real and fake
            # Note the tensor shape of eps - the three dimensionswill be broadcasted to channel, width, height of images
            eps = torch.rand(batch_size, 1, 1, 1, device=device)
            x_hat = eps * images + (1 - eps) * G
            x_hat.requires_grad_(True)

            D_hat = model.discriminator(x_hat)
            grad_outputs = torch.ones_like(D_hat)

            gradients = torch.autograd.grad(
                outputs=D_hat,
                inputs=x_hat,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]

            grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
            gradient_penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()

            # Total discriminator loss
            S_real = torch.mean(model.discriminator(images))
            S_fake = torch.mean(model.discriminator(G))
            loss_D = -(S_real-S_fake) + gradient_penalty
            loss_D.backward()
            discriminator_optimizer.step()

            discriminator_loss += loss_D.item()
            score_real += S_real.item()
            score_fake += S_fake.item()

            ########################################################################
            # Step 2. Train the generator, keep the discriminator fixed.
            ########################################################################
            model.discriminator.eval()
            model.generator.train()
            generator_optimizer.zero_grad()

            # sample a fresh batch of latent vectors, since we want to update the generator based on new samples
            latent_vectors = torch.randn(batch_size, latent_dim, device=device)
            G = model.generator(latent_vectors)

            # Generator tries to fool discriminator, so has to increase the average score (=minimize negative score)
            loss_G = -torch.mean(model.discriminator(G))
            loss_G.backward()

            generator_optimizer.step()
            generator_loss += loss_G.item()

        # Log a checkpoint every 10 epochs
        model_id = None
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # Log model checkpoint with step parameter
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f"model-checkpoint-{epoch}",
                step=epoch
            )
            print(f"Epoch {epoch}, D Loss: {discriminator_loss:.4f} G Loss: {generator_loss:.4f}")
            model_id = model_info.model_id

        discriminator_loss /= len(train_loader)
        generator_loss /= len(train_loader)
        score_real /= len(train_loader)
        score_fake /= len(train_loader)

        mlflow.log_metrics(
            {
                "discriminator_loss": discriminator_loss,
                "generator_loss": generator_loss,
                "score_real": score_real,
                "score_fake": score_fake,
                "lr_g": generator_optimizer.param_groups[0]['lr'],
                "lr_d": discriminator_optimizer.param_groups[0]['lr'],
            },
            step=epoch,
            model_id=model_id,
        )

    print(f'Training complete in {time.time()-start:.4f}s.')
    return model_id
