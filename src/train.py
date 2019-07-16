import os
import random
random.seed(123)
import torch
import torch.optim as optim

from network import Siamese
from loss import ContrastiveLoss
from dataloader import mnist

if __name__=="__main__":
    # load dataset
    trainloader, testloader, classes, color = mnist()

    # I don't know good margin value.
    net = Siamese(nItem=28*28, mid_dim=2)
    criterion = ContrastiveLoss(margin=2)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train network
    for epoch in range(3):
        running_loss=0.0
        for i, data in enumerate(trainloader):
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
    torch.save(net.state_dict(), "./net/siamese.pt")

    # load network
    # net.load_state_dict(torch.load("./net/siamese.pt"))

