import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from numpy.random import uniform


'''modelの定義'''
'''Discriminater'''
class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.main = nn.Sequential(
            #input is nb x 1 x 28 x 28
            nn.Linear(1*28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 1*28*28)
        x = self.main(x)
        #x = nn.parallel.data_parallel(self.main, x, range(3))
        return x

'''Generator'''
class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()
        self.main = nn.Sequential(
            #input Z is nb x 100 normalize(0, 1)
            nn.Linear(100,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1*28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x.view(bs,100))
        #x = nn.parallel.data_parallel(self.main, x, range(3))
        return x.view(-1, 1, 28, 28)


'''Loss関数の定義'''
criteion = nn.BCELoss()
criteion.cuda()

'''パラメータの設定'''
niter = 30000
bs = 1000


input = torch.cuda.FloatTensor(bs, 1, 28, 28)
noise = torch.cuda.FloatTensor(uniform(0,1,(bs, 100, 1, 1))) #npの一様分布からノイズを作る
fixed_noise = torch.cuda.FloatTensor(bs, 100, 1, 1).normal_(0, 1)
label = torch.cuda.FloatTensor(bs)

real_label = 1
fake_label = 0

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)



net_D = netD()
net_D.cuda()
net_G = netG()
net_G.cuda()

'''optimizerの定義'''
optimizerD = optim.Adam(net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(net_G.parameters(), lr=0.0001, betas=(0.5, 0.999))



'''学習データの準備(MNISTデータ)'''
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('MNIST_data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=bs
)



'''学習'''
for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        '''net_D 最大化 log(D(x)) + log(1 - D(G(z)))'''
        '''train with real (data)'''
        net_D.zero_grad()
        real, _ = data
        input.data.resize_(real.size()).copy_(real)
        label.data.resize_(bs).fill_(real_label)

        output = net_D(input)
        errD_real = criteion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        '''train with fake (generated)'''
        noise.data.resize_(bs, 100, 1, 1)
        noise.data.normal_(0, 1)
        fake = net_G(noise)
        label.data.fill_(fake_label)
        output = net_D(fake.detach())
        errD_fake = criteion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        '''net_G 最大化 log(D(G(z)))'''
        net_G.zero_grad()
        label.data.fill_(real_label)
        output = net_D(fake)
        errG = criteion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                 % (epoch, niter, i, len(dataloader),
                   errD.data[0], errG.data[0],  D_x, D_G_z1, D_G_z2))
    if epoch % 10 == 0:
        fake = net_G(fixed_noise)
        vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'
                              % ('output', epoch),normalize=True)
