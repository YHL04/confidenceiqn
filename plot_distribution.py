

import matplotlib.pyplot as plt
import seaborn as sns


def plot_image_distribution(image, model, conf):
    pred = model(image)
    pred_error, taus = conf(image, n_tau=4096)

    pred_error = pred_error.squeeze().detach().numpy()
    plot_distribution(pred_error)


def plot_distribution(taus):
    sns.displot(taus)
    plt.show()


if __name__ == "__main__":
    from model import Net, ConfModel
    from dataloader import get_MNIST_dataloaders

    trainloader, testloader = get_MNIST_dataloaders(batch_size=1, test_batch_size=1)

    dim_x, dim_y, channels, classes = 28, 28, 1, 10

    model = Net(dim_x=dim_x,
                dim_y=dim_y,
                channels=channels,
                classes=classes)
    conf = ConfModel(dim_x=dim_x,
                     dim_y=dim_y,
                     channels=channels)

    for data, target in testloader:
        plot_image_distribution(data, model, conf)

