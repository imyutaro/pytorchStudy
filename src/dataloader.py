from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def mnist():
    """
    I don't know why mean=0,1307 and std=0.3081.
    See https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    """
    mean, std = 0.1307, 0.3081

    train_dataset = MNIST('../data/MNIST', train=True, download=True,
                          transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))]))

    test_dataset = MNIST('../data/MNIST', train=False, download=True,
                         transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((mean,), (std,))]))

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    return train_dataset, test_dataset, classes, colors


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.save("tmp.png")
    

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


# function to show img
def imshow(img, i):
    npimg = img.numpy().squeeze()
    # plt.imshow(npimg)
    plt.savefig("tmp"+str(i)+".png")


if __name__=="__main__":

    train_dataset, test_dataset, classes, colors = mnist()

    for i, data in enumerate(train_dataset):
        inputs, label = data
        imshow(inputs, i)
        if i==3:
            exit(0)

