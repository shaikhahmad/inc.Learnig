from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable

# Writing our model
class NewAlexnet(nn.Module):
    def __init__(self, alexnet):
        super().__init__()

        self.add_module('features', alexnet.features)
        self.add_module('classifier', alexnet.classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.weights = []

    def __apweighta__(self, features):
        flatweights = torch.Tensor([])
        for i in range(len(features)):
            if hasattr(features[i], 'weight'):
                x = features[i].weight.data.view(-1).clone()
                x = x.cpu()
                flatweights = torch.cat((flatweights, x), 0)
        return flatweights.clone()

    def currentWeights(self):
        flatweights = torch.Tensor([])
        for m in self.features:
            if hasattr(m, 'weight'):
                x = m.weight.data.view(-1).clone()
                x = x.cpu()
                flatweights = torch.cat((flatweights, x), 0)
        return flatweights

    def setmyweights(self, weight):
        x = weight.clone()
        for m in self.features:
            if hasattr(m, 'weight'):
                sp = torch.Tensor(list(m.weight.shape))
                ind = int(torch.prod(sp))
                y = x[:ind]
                m.weight.data = y.clone().view(m.weight.shape)
                x = x[ind:]

    def forward(self, x, train=True):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
#         if train:
#             # self.weights.append(self.classifier.weights.view(-1))
#             self.weights.append(self.__apweighta__(self.features))
        return x

class NewLeNet(nn.Module):
    def __init__(self, ):
        super(NewLeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        self.weights = []

    def __apweighta__(self, features):
        flatweights = torch.Tensor([])
        for i in range(len(features)):
            if hasattr(features[i], 'weight'):
                x = features[i].weight.data.view(-1).clone()
                x = x.cpu()
                flatweights = torch.cat((flatweights, x), 0)
        return flatweights.clone()

    def currentWeights(self):
        flatweights = torch.Tensor([])
        for m in self.features:
            if hasattr(m, 'weight'):
                x = m.weight.data.view(-1).clone()
                x = x.cpu()
                flatweights = torch.cat((flatweights, x), 0)
        return flatweights

    def setmyweights(self, weight):
        x = weight.clone()
        for m in self.features:
            if hasattr(m, 'weight'):
                sp = torch.Tensor(list(m.weight.shape))
                ind = int(torch.prod(sp))
                y = x[:ind]
                m.weight.data = y.clone().view(m.weight.shape)
                x = x[ind:]

    def forward(self, x, train=True):
        x = self.features(x)
        # x = torch.flatten(x, 1)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

class autoencoder(nn.Module):
    def __init__(self, in_dim, h_dim=100):
        super(autoencoder, self).__init__()
        mid_dim = (in_dim + h_dim)//2
        self.encode = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, h_dim),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.Linear(h_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, in_dim),
            nn.ReLU()
        )

    def forward(self, x):
        h = self.encode(x)
        recon = self.decode(h)
        return recon


class VAE(nn.Module):
    def __init__(self, in_dim, h_dim=100):
        super(VAE, self).__init__()
        mid_dim = in_dim // 3

        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, mid_dim)

        self.fc_mean = nn.Linear(mid_dim, h_dim)
        self.fc_std = nn.Linear(mid_dim, h_dim)

        self.fc3 = nn.Linear(h_dim, mid_dim)
        self.fc4 = nn.Linear(mid_dim, mid_dim)
        self.fc5 = nn.Linear(mid_dim, mid_dim)
        self.fc6 = nn.Linear(mid_dim, in_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc_mean(x), self.fc_std(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.tanh(self.fc3(z))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        return self.sigmoid(self.fc6(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar







# class VAE(nn.Module):
#     def __init__(self, in_dim, h_dim=100):
#         super(VAE, self).__init__()
#
#         self.fc11 = nn.Linear(in_dim, h_dim)
#         self.fc12 = nn.Linear(in_dim, h_dim)
#         self.fc2 = nn.Linear(h_dim, in_dim)
#
#         # self.activation = nn.ReLU(inplace=True)
#
#     def encode(self, x):
#         h1 = nn.ReLU()(x)
#         return self.fc11(x), self.fc12(x)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std
#
#     def decode(self, z):
#         h3 = nn.ReLU(inplace=True)(z)
#         return nn.Sigmoid()(self.fc2(h3))
#
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# class AlexNet(nn.Module):







# import torchvision
# import torchvision.transforms as transforms
#
# transform = transforms.Compose(
#         [transforms.ToTensor(), ])
#
# trainset = torchvision.datasets.MNIST(root='/home/ahmad/th_v2/data', train=True,
#                                             download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
#                                           shuffle=True, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#
#
# import torchvision.models as models
# alexnet = models.alexnet(pretrained=True)
#
#
# dim = 100
# encoder = Autoencoder(784,dim)
#
# if torch.cuda.is_available():
#     encoder.cuda()
#
#
#
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
# iter_per_epoch = len(trainloader)
# data_iter = iter(trainloader)
#
# # save fixed inputs for debugging
# fixed_x, _ = next(data_iter)
# # torchvision.utils.save_image(Variable(fixed_x).data.cpu(), './data/real_images.png')
# fixed_x = fixed_x.view(fixed_x.size(0), -1).cuda()
#
# num_epochs = 10
# for epoch in range(num_epochs):
#     for i, (images, _) in enumerate(trainloader):
#
#         # flatten the image
#         images = images.view(images.size(0), -1).cuda()
#         out = encoder(images)
#         loss = criterion(out, images)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i+1) % 100 == 0:
#             print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f '
#                 %(epoch+1, num_epochs, i+1, len(trainset)//100, loss.data))
#
#     # save the reconstructed images
#     reconst_images = encoder(fixed_x)
#     reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
#     # torchvision.utils.save_image(reconst_images.data.cpu(), './data/reconst_images_%d.png' % (epoch+1))
