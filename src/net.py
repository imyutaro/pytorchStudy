import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    # In init I have to write layer which has learnable parameters.
    def __init__(self):
        super(Net, self).__init__()

        # nn.Conv2d(input_dim, output_dim, 
        #     (filter_width, filter_height),
        #     (stride_width, stride_height),
        #     (padding_width, padding_height)
        #   )
        # if you write like below, filter, stride and padding width and height are equal.
        # nn.Conv2d(1, 6, 4, 5, 2)

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # In forward I have to write how to calculate forward pass.
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 


if __name__=="__main__":

    import numpy as np
    # make a neural network instance
    net = Net()
    print(net)

    # get params and print
    params = list(net.parameters())
    print(len(params))

    for i in range(len(params)):
        print(params[i].size())

    # pytorch only supports minibatch,
    # so Conv2d input has to be nSamples * nChannels * Height * Width.
    # If I have a single sample, use input.unsqueeze(0) to add a fake batch dimension.
    input = torch.randn(1, 1, 32, 32)
    print("no unsqueeze\n", input.size())
    input = torch.randn(1, 32, 32).unsqueeze(0)
    print("unsqueeze\n", input.size())
    output = net(input)
    print(output)
    
    target = torch.randn(10).view(1, -1) 
    # => shape will be (1, 10) to be same as output shape from net.

    # Set criterion(loss function)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(f"loss: {loss}")

    # Set zeros to the gradient buffers of all parameters
    net.zero_grad()

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)


    # To make sure substraction of weight - grad * learning_rate
    # for name, param in net.named_parameters():
    #     if param.requires_grad and name == "conv1.bias":
    #         print(name)
    #         print(param.data)
    #
    # learning_rate = 1
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)

    # for name, param in net.named_parameters():
    #     if param.requires_grad and name == "conv1.bias":
    #         print(name)
    #         print(param.data)


    # Instead of above code, pytorch has optim method to update weight.
    import torch.optim as optim 

    # create optimizer
    optimizer = optim.SGD(net.parameters(), lr=1)
    # training loop
    # Set zeros to the gradient buffers of all parameters
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    # Does update
    optimizer.step() 

