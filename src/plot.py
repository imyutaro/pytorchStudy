import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm
import torch

from network import Siamese, ConvSiamese
from dataloader import mnist


if __name__=="__main__":

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
