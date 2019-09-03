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
        for epoch in range(epochs):
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


def assoc_siamese(filename, train=False, epochs=3):
    # load dataset
    data = assoc(filename)

    # define network
    net = Siamese(nItem=data.item_len(), mid_dim=2)

    if train:
        # I don't know good margin value.
        criterion = ContrastiveLoss(margin=2)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        transactions = data.get_trans() 

        count = 0
        # train network
        for epoch in range(epochs):
            for trans in transactions:
                for i, item1 in enumerate(trans):
                    time_1 = time.time()
                    t_sub = trans[i+1:]

                    # get data
                    input1 = torch.from_numpy(data[item1]).float()
                    time_2 = time.time()
                    for item2 in t_sub:
                        # initialize grad to 0.
                        optimizer.zero_grad()

                        input2 = torch.from_numpy(data[item2]).float()
                        output1, output2 = net(input1, input2)
                        time_3 = time.time()

                        loss = criterion(output1, output2, t=1)
                        loss.backward()
                        time_4 = time.time()
                        optimizer.step()
                        time_5 = time.time()
                        print(f"1-2: {time_2 - time_1}")
                        print(f"2-3: {time_3 - time_2}")
                        print(f"3-4: {time_4 - time_3}")
                        print(f"4-5: {time_5 - time_4}")

            if count%1000==0:
                print("Finished {count} transactions")

            os.makedirs("net", exist_ok=True)
            torch.save(net.state_dict(), f"./net/assoc_SiameseEpoch{epoch}.pt")
            print(f"Finish epoch{epoch}")

        print("Finish training")
    else:
        # load network
        net.load_state_dict(torch.load("./net/assoc_Siamese.pt"))

    return net


if __name__=="__main__":
    assoc_siamese(filename="retail", train=True, epochs=1)

