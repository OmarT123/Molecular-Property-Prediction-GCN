import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_loss(*args, **kwargs):
    """Plot the loss for each epoch

    Args:
        epochs (int): number of epochs
        train_loss (array): training losses for each epoch
        val_loss (array): validation losses for each epoch
    """
    save_path = 'trash.png'
    title = "Model Loss"
    for key, value in kwargs.items():
        if key == 'save_path':
            save_path=value
            continue
        if key == 'title':
            title = value
            continue
        plt.plot(value, label=key)

    # plt.plot(gcn_train_loss, label="MSE")
    # plt.plot(gcn_val_loss, label="MAE")
    plt.legend()
    # plt.ylabel("loss")
    plt.xlabel(args[0])
    plt.title(title)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    plt.savefig(save_path)