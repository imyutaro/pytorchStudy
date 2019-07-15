import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm
import torch

from network import Siamese
from dataloader import mnist


if __name__=="__main__":

    # load dataset
    trainloader, testloader, classes, color = mnist()

    # load network
    net = Siamese(nItem=28*28, mid_dim=2)
    net.load_state_dict(torch.load("./net/siamese.pt"))

    # plot embedding vector
    with torch.no_grad():
        fig = plt.figure()
        ax = plt.axes()
        for i, data in tqdm(enumerate(trainloader)):
            x, label = data
            y = net.get_embedding(x.reshape(-1)).numpy()

            plt.scatter(x=y[0], y=y[1], color=color[label])

    plt.savefig("output.png")

