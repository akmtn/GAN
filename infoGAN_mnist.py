from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#ngpuは１で固定するようにしている
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# gaussian_nllはchainerのコードを参考に
def gaussian_nll(x, mean, ln_var):
    x_prec = torch.exp(-ln_var).cuda()
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5

    loss = (ln_var + torch.log(2 * Variable(torch.cuda.FloatTensor([[np.pi, np.pi]]*100)))) / 2 - x_power
    #以下，この総和の取り方があっているのか不明
    loss = torch.sum(loss, 0) / opt.batchSize
    loss = torch.sum(loss) / 2
    return loss


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()

        # input is Z, going into a convolution
        self.fc1 = nn.Linear(74, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128 * 7 * 7)
        self.fc2_bn = nn.BatchNorm1d(128 * 7 * 7)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False) #(bs, 64, 14, 14)
        self.upconv1_bn = nn.BatchNorm2d(64)
        self.upconv2 = nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False) #(bs, 1, 28, 28)


    def forward(self, input):
        x = F.relu(self.fc1_bn(self.fc1(input)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = x.view(opt.batchSize, 128, 7, 7)
        x = F.relu(self.upconv1_bn(self.upconv1(x)))
        x = self.upconv2(x)

        return x

netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        # input is (bs, 1 channel, 28, 28)
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        # state size. (bs, 64, 14, 14)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(128)
        # state size. (bs, 128 , 7, 7)
        #fullchainに繋ぐ
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        #Discriminator
        self.fcd = nn.Linear(1024, 2)
        self.softmax = nn.Softmax()
        #Q network
        self.fcq1 = nn.Linear(1024, 128)
        self.fcq1_bn = nn.BatchNorm1d(128)
        self.fcq2 = nn.Linear(128, 12)


    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.1)
        x = x.view(opt.batchSize, -1)
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)), 0.1) # -> (bs, 1024)

        #discriminator D
        out_d = self.softmax(self.fcd(x))

        #recognition network Q
        out_q = F.leaky_relu(self.fcq1_bn(self.fcq1(x)), 0.1)
        out_q = self.fcq2(out_q)

        return out_d, out_q


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


input = torch.FloatTensor(opt.batchSize, 1, 28, 28)
noise = torch.FloatTensor(opt.batchSize, 74, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, 74, 1, 1).normal_(0, 1)
label = torch.LongTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda: #このコードはcudaを使うことを前提にしている
    netD.cuda()
    netG.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


'''学習データの準備(MNISTデータ)'''
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('MNIST_data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=opt.batchSize
)


for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # z: 10-dim, unif(-1, 1)
        z = np.random.uniform(-1, 1, (opt.batchSize, 62)).astype(np.float32)

        # category c
        category_labels = np.random.randint(10, size=opt.batchSize)
        one_hot = np.zeros((opt.batchSize, 10))
        one_hot[np.arange(opt.batchSize), category_labels] = 1
        c_categorical = np.asarray(one_hot, dtype=np.float32)
        category_labels = np.asarray(category_labels, dtype=np.int64)
        category_labels = Variable(torch.from_numpy(category_labels).cuda())

        #continuous c
        c_continuous = np.random.rand(opt.batchSize, 2) * 2 - 1
        c_continuous = np.asarray(c_continuous, dtype=np.float32)

        #noise : concatenate( z , categorical c, continuous c)
        zc = np.concatenate((z, c_categorical, c_continuous), axis=1)
        noise.data.copy_(torch.from_numpy(zc))
        c_continuous = Variable(torch.from_numpy(c_continuous).cuda())

        #
        netD.zero_grad()
        netG.zero_grad()

        # D real forward
        x_real, _ = data
        input.data.copy_(x_real)
        y_real, _ = netD(input)
        # D real backward
        errD_real = F.cross_entropy(y_real, label)
        errD_real.backward()

        # G generate x
        x_fake = netG(noise.view(opt.batchSize,74))

        # D fake forward
        y_fake, mi = netD(x_fake.detach())
        # D fake backward
        label.data.fill_(fake_label)
        errD_fake = F.cross_entropy(y_fake, label)
        errD_fake.backward()

        errD = errD_real + errD_fake

        # Gで作られたx_fakeをDに入れる．出力はy_fake
        label.data.fill_(real_label)
        y_fake, mi = netD(x_fake)
        errG = F.cross_entropy(y_fake, label)

        #mutual information mi
        #categorical_loss
        mi_categorical, mi_continuous_mean = torch.split(mi, 10, dim=1)
        categorical_loss = F.cross_entropy(mi_categorical, category_labels)
        #continuous_loss
        mi_continuous_ln_var = torch.cuda.FloatTensor([[1, 1]]*opt.batchSize)
        continuous_loss = gaussian_nll(mi_continuous_mean, c_continuous, Variable(mi_continuous_ln_var))

        # G loss, backward
        errG += categorical_loss
        errG += continuous_loss
        errG.backward()


        #更新
        optimizerD.step()
        optimizerG.step()


        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.data[0], errG.data[0]))
    if epoch % 10 == 0:
        vutils.save_image(x_real,
                    'out/real_samples.png',
                    normalize=True)
        fake = netG(noise.view(opt.batchSize,74))
        vutils.save_image(fake.data,
                'out/fake_samples_epoch_%03d.png' % (epoch),
                normalize=True)
    if epoch % 100 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), 'out/netG_epoch_%d.pth' % epoch)
        torch.save(netD.state_dict(), 'out/netD_epoch_%d.pth' % epoch)
