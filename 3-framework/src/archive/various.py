# original version for list of models
def print_accuracies(models, test_loader = test_loader):
    for model_name, model in models.items():
        print(f"** Model: {model_name}")
        model.to('cpu')
        model.eval()
        get_accuracy(model, test_loader)

# original version for list of models
def plot_losses(train_loss, val_loss, models):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, model in enumerate(models):
        t_loss = train_loss[model]
        v_loss = val_loss[model]
        n = len(t_loss)
        assert len(t_loss) == len(v_loss)
        plt.plot(range(n), t_loss, label=f'Train {model}', color=colors[i])
        plt.plot(range(n), v_loss, '--', label=f'Val {model}', color=colors[i])
    plt.legend()
    plt.ylabel("Loss")
    plt.savefig('../plots/loss.png')


# new version for single model
def plot_losses(train_loss, val_loss, model):
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label=f'Train')
    plt.plot(range(epochs), val_loss,   label=f'Val')
    plt.legend()
    plt.ylabel("Loss")
    plt.savefig('../plots/loss.png')

plot_losses(training_loss, validation_loss, resnet)