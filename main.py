

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

from model import Net, ConfModel
from dataloader import get_MNIST_dataloaders, get_CIFAR10_dataloaders, get_CIFAR100_dataloaders


def train(model, train_loader, optimizer, epoch):
    loss_list = []

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        loss_list.append(loss.item())

    return loss_list


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def quantile_loss(expected, target, taus):
    batch_size = expected.size(0)
    n_tau = expected.size(1)

    assert expected.shape == (batch_size, n_tau)
    assert target.shape == (batch_size,)
    assert taus.shape == (batch_size, n_tau)
    assert not taus.requires_grad

    batch_size = expected.size(0)

    expected = expected.view(batch_size, n_tau, 1)
    target = target.unsqueeze(1).repeat(1, n_tau).unsqueeze(1)
    taus = taus.view(batch_size, n_tau, 1)

    assert target.shape == (batch_size, 1, n_tau)

    td_error = target - expected
    huber_loss = torch.where(td_error.abs() <= 1, 0.5 * td_error.pow(2), td_error.abs() - 0.5)
    quantile_loss = abs(taus - (td_error.detach() < 0).float()) * huber_loss

    loss = quantile_loss.sum(dim=1).mean(dim=1)
    loss = loss.mean()

    return loss


def train_conf(model, conf, trainloader, optimizer, epoch):
    model.eval()
    conf.train()
    losses = []

    count = 0
    error_mean = 0
    error_std = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()

        pred = model(data)

        error = F.nll_loss(pred, target, reduction="none")

        # recalculate error
        # count += 1
        #
        # error_mean = ((count - 1)/count) * error_mean + \
        #              (error.mean() / count).item()
        #
        # error_std = ((count - 1)/count) * error_std + \
        #             (torch.pow(error - error_mean, 2).mean() / count).item()

        # tracking error mean and std, and normalizing error
        # error = error - error_mean
        # error = error / error_std

        # print("error mean", error_mean)
        # print("error ", error.mean())

        pred_error, taus = conf(data, n_tau=64)

        loss = quantile_loss(pred_error, error, taus)
        # loss = F.mse_loss(pred_error, error)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(conf.parameters(), 1)
        optimizer.step()

        if batch_idx % 100 == 0:

            print('Conf Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))

        losses.append(loss.item())

    return losses


def val_conf(model, conf, dataloader):

    num_wrong = 0
    avg_uncertainty_wrong = 0
    num_right = 0
    avg_uncertainty_right = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.cuda()
        target = target.cuda()

        with torch.no_grad():
            pred = model(data)
            pred = torch.argmax(pred, dim=1)

            pred_error, taus = conf(data, n_tau=64)
            pred_error = pred_error.mean(dim=1)

            for i in range(len(pred)):
                if pred[i] == target[i]:
                    num_right += 1
                    avg_uncertainty_right += pred_error[i].item()
                else:
                    num_wrong += 1
                    avg_uncertainty_wrong += pred_error[i].item()

        # print("pred {} real {} estimated error {}"
        #       .format(pred[0], target[0], pred_error[0]))

    zero_data = torch.zeros(1, 28, 28)

    pred_error, taus = conf(data, n_tau=64)
    pred_error = pred_error.mean()

    print("zero data estimated error ", pred_error.item())

    print("average uncertainty of wrong ", avg_uncertainty_wrong / num_wrong)
    print("average uncertainty of right ", avg_uncertainty_right / num_right)


def main(epochs=20, conf_epochs=20, dataset="cifar100"):

    if dataset == "mnist":
        dim_x, dim_y, channels, classes = 28, 28, 1, 10
        trainloader, testloader = get_MNIST_dataloaders(batch_size=32, test_batch_size=32)

    if dataset == "cifar10":
        dim_x, dim_y, channels, classes = 32, 32, 3, 10
        trainloader, testloader = get_CIFAR10_dataloaders(batch_size=32, test_batch_size=32)

    if dataset == "cifar100":
        dim_x, dim_y, channels, classes = 32, 32, 3, 100
        trainloader, testloader = get_CIFAR100_dataloaders(batch_size=32, test_batch_size=32)

    model = Net(dim_x=dim_x,
                dim_y=dim_y,
                channels=channels,
                classes=classes).cuda()
    conf = ConfModel(dim_x=dim_x,
                     dim_y=dim_y,
                     channels=channels).cuda()

    opt_model = optim.Adadelta(model.parameters(), lr=1.0)
    opt_conf = optim.Adadelta(conf.parameters(), lr=1.0)

    # opt_model = optim.Adam(model.parameters(), lr=1e-4)
    # opt_conf = optim.Adam(conf.parameters(), lr=1e-4)

    scheduler_model = StepLR(opt_model, step_size=1, gamma=0.7)
    scheduler_conf = StepLR(opt_model, step_size=1, gamma=0.7)

    losses = []
    for epoch in range(epochs):
        loss = train(model, trainloader, opt_model, epoch)
        losses.extend(loss)

        test(model, testloader)

        scheduler_model.step()

    conf.load_state_dict(model.state_dict(), strict=False)

    conf_losses = []
    for epoch in range(conf_epochs):
        conf_loss = train_conf(model, conf, trainloader, opt_conf, epoch)
        conf_losses.extend(conf_loss)

        scheduler_conf.step()

    val_conf(model, conf, trainloader)
    val_conf(model, conf, testloader)

    plt.subplot(2, 1, 1)
    sns.lineplot(data=losses)

    plt.subplot(2, 1, 2)
    sns.lineplot(data=conf_losses)

    plt.show()


if __name__ == "__main__":
    main()

