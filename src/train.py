import os
import random
random.seed(123)
import torch
import torch.optim as optim

from network import Siamese, ConvSiamese
from loss import ContrastiveLoss
from dataloader import mnist, assoc


def mnist_siamese(train=False, epochs=3):
    # load dataset
    trainloader, testloader, classes, color = mnist()

    # define network
    net = Siamese(nItem=28*28, mid_dim=2)

    if train:
        # I don't know good margin value.
        criterion = ContrastiveLoss(margin=2)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # train network
        for epoch in range(3):
            running_loss=0.0
            for i, data in enumerate(trainloader):
                random.seed(random.randint(0, 3000))

                optimizer.zero_grad()

                input1, label1 = data
                input2, label2 = trainloader[random.randint(0, len(trainloader)-1)]
                input1 = input1.view(-1)
                input2 = input2.view(-1)
                output1, output2 = net(input1, input2)
                t= 1 if (label1==label2) else 0

                loss = criterion(output1, output2, t)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i%2000==1999:
                    print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, running_loss/2000))
                    running_loss = 0.0

        print("Finish training")
        os.makedirs("net", exist_ok=True)
        torch.save(net.state_dict(), "./net/Siamese.pt")
    else:
        # load network
        net.load_state_dict(torch.load("./net/Siamese.pt"))

    return net



def mnist_convsiamese(train=False, epochs=3):
    # load dataset
    trainloader, testloader, classes, color = mnist()

    # define network
    net = ConvSiamese()

    if train:
        # I don't know good margin value.
        criterion = ContrastiveLoss(margin=1)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # train network
        for epoch in range(epochs):
            running_loss=0.0
            for i, data in enumerate(trainloader):
                random.seed(random.randint(0, 3000))

                optimizer.zero_grad()

                input1, label1 = data
                input2, label2 = trainloader[random.randint(0, len(trainloader)-1)]
                input1 = input1.unsqueeze(dim=0)
                input2 = input2.unsqueeze(dim=0)
                output1, output2 = net(input1, input2)
                t= 1 if (label1==label2) else 0

                loss = criterion(output1, output2, t)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i%2000==1999:
                    print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, running_loss/2000))
                    running_loss = 0.0

        print("Finish training")
        os.makedirs("net", exist_ok=True)
        torch.save(net.state_dict(), "./net/convSiamese.pt")
    else:
        # load network
        net.load_state_dict(torch.load("./net/convSiamese.pt"))

    return net


def assoc_siamese(filename, train=False, epoch=3):
    # load dataset
    data = assoc(filename)

    # define network
    net = Siamese(nItem=data.item_len(), mid_dim=2)

    if train:
        # I don't know good margin value.
        criterion = ContrastiveLoss(margin=2)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # train network
        for epoch in range(3):
            for t in data.get_trans():
                for i in range(len(t)):
                    # get data
                    item1 = t[i]
                    input1 = torch.from_numpy(data[item1]).float()
                    for j in range(i+1, len(t)):
                        # initialize grad to 0.
                        optimizer.zero_grad()

                        item2 = t[j]
                        input2 = torch.from_numpy(data[item2]).float()
                        output1, output2 = net(input1, input2)

                        loss = criterion(output1, output2, t=1)
                        loss.backward()
                        optimizer.step()

        print("Finish training")
        os.makedirs("net", exist_ok=True)
        torch.save(net.state_dict(), "./net/assoc_Siamese.pt")
    else:
        # load network
        net.load_state_dict(torch.load("./net/assoc_Siamese.pt"))

    return net


if __name__=="__main__":
    assoc_siamese(filename="retail", train=True)

