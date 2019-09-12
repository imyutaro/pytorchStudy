import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm
import torch

from network import Siamese, ConvSiamese
from dataloader import mnist


def plot_convsiamese():
    # load dataset
    trainloader, testloader, classes, color = mnist()

    """
    # load network
    net = Siamese(nItem=28*28, mid_dim=2)
    net.load_state_dict(torch.load("./net/Siamese.pt"))

    # plot embedding vector
    with torch.no_grad():
        fig = plt.figure()
        ax = plt.axes()
        for i, data in enumerate(tqdm(trainloader)):
            x, label = data
            y = net.get_embedding(x.view(-1)).numpy()

            plt.scatter(x=y[0], y=y[1], color=color[label])

    plt.legend(classes)
    plt.savefig("output.png")
    """
    # load network
    net = ConvSiamese()
    net.load_state_dict(torch.load("./net/convSiamese.pt"))
    print(color, classes)
    exit(0)

    # plot embedding vector
    with torch.no_grad():
        fig = plt.figure()
        ax = plt.axes()
        for i, data in enumerate(tqdm(trainloader)):
            x, label = data
            x = x.unsqueeze(dim=0)
            y = net.get_embedding(x).numpy()

            ax.scatter(x=y[0], y=y[1], alpha=0.5, color=color[label], label=classes[label])

    ax.legend()
    plt.savefig("output.png")


def plot_assoc_siamese():
    # load dataset
    filename = "T10I4D100K"
    data = assoc(filename)

    # load network
    net = Siamese(nItem=data.item_len(), mid_dim=2)
    trained_model = os.path.join("./net", filename, "2019-09-11", "assoc_SiameseEpoch1.pt")
    net.load_state_dict(torch.load(trained_model))
    items = list(data.item)

    # plot embedding vector
    with torch.no_grad():
        fig = plt.figure()
        ax = plt.axes()
        for i, item in enumerate(tqdm(items)):
            x = torch.from_numpy(data[item]).float()
            x = x.unsqueeze(dim=0)
            y = net.get_embedding(x).numpy()[0]

            ax.scatter(x=y[0], y=y[1], alpha=0.5)

    ax.legend()
    plt.savefig("tmp.png")

if __name__=="__main__":
    plot_assoc_siamese()
