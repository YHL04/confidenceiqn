

import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_image_distribution(image, model, conf, mean):
    image = image.cuda()
    pred = model(image)
    pred_error, taus = conf(image, n_tau=10000)

    pred_error = pred_error.cpu().squeeze().detach().numpy()
    image = image.cpu().squeeze().detach().numpy()
    plot_distribution(pred_error, image, mean)


def plot_distribution(taus, image, mean):

    # plot image
    plt.imshow(image, cmap='gray')

    # plot distribution
    sns.displot(taus)
    plt.axvline(x=mean, c='red')
    plt.xlim(xmin=-0.1, xmax=0.1)
    plt.show()


def get_mean(dataloader, conf):
    error = []
    for data, target in dataloader:
        data = data.cuda()
        pred_error, taus = conf(data, n_tau=64)
        error.append(pred_error)

    mean = torch.stack(error).mean()
    return mean.cpu().squeeze().detach().numpy()


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

    model.load_state_dict(torch.load("saved/model_mnist"))
    conf.load_state_dict(torch.load("saved/conf_mnist"))

    model.cuda()
    conf.cuda()

    mean = get_mean(testloader, conf)
    print(mean)

    for data, target in testloader:
        plot_image_distribution(data, model, conf, mean)

