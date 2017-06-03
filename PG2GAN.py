
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--niterG1', type=int, default=10, help='number of epochs to train for G1')
parser.add_argument('--niterG2', type=int, default=10, help='number of epochs to train for G2')
parser.add_argument('--L1_lambda', type=int, default=1, help='L1_lambda for G2')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)
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


'''Generator at stage-1 (G1)'''
class _netG1(nn.Module):
    def __init__(self):
        super(_netG1, self).__init__()
        # input bs x 21 x 256 x256
        self.conv_1 = nn.Conv2d(21, 64, kernel_size=3, stride=1, padding=1)
        # state bs x 64 x 256 x 256
        self.e_block_1 = ResBlock(64)
        self.conv_2 = nn.Conv2d(64, 128, 3, 2, 1)
        #state bs x 128 x 128 x128
        self.e_block_2 = ResBlock(128)
        self.conv_3 = nn.Conv2d(128, 256, 3, 2, 1)
        #state bs x 256 x 64 x64
        self.e_block_3 = ResBlock(256)
        self.conv_4 = nn.Conv2d(256, 384, 3, 2, 1)
        #state bs x 384 x 32 x32
        self.e_block_4 = ResBlock(384)
        self.conv_5 = nn.Conv2d(384, 512, 3, 2, 1)
        #state bs x 512 x 16 x16
        self.e_block_5 = ResBlock(512)
        self.conv_6 = nn.Conv2d(512, 640, 3, 2, 1)
        #state bs x 640 x 8 x 8
        self.e_block_6 = ResBlock(640)
        self.conv_7 = nn.Conv2d(640, 640, 3, 1, 1)
        #state bs x 1280 x 8 x 8
        # have to do view
        self.fc_1 = nn.Linear(640*8*8, 64)
        self.fc_2 = nn.Linear(64, 640*8*8)
        # have to do view
        #state bs x 1280 x 8x 8
        self.de_block_1 = ResBlock(640)
        self.deconv_1 = nn.ConvTranspose2d(640, 512, kernel_size=3, stride=2, padding=1)
        # state bs x 640 x 16 x 16
        self.de_block_2 = ResBlock(512)
        self.deconv_2 = nn.ConvTranspose2d(512, 384, kernel_size=3, stride=2, padding=1)
        #state bs x 512 x 32 x 32
        self.de_block_3 = ResBlock(384)
        self.deconv_3 = nn.ConvTranspose2d(384, 256,kernel_size=3, stride=2, padding=1)
        #state bs x 384 x 64 x64
        self.de_block_4 = ResBlock(256)
        self.deconv_4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        #state bs x 256 x 128 x 128
        self.de_block_5 = ResBlock(128)
        self.deconv_5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        #state bs x 128 x 256 x 256
        self.de_block_6 = ResBlock(64)
        self.deconv_6 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        #state bs x 3 x 256 x 256


    def forward(self, input):
        # encoding
        out_from_e_1 = self.e_block_1(self.conv_1(input))
        out_from_e_2 = self.e_block_2(self.conv_2(out_from_e_1))
        out_from_e_3 = self.e_block_3(self.conv_3(out_from_e_2))
        out_from_e_4 = self.e_block_4(self.conv_4(out_from_e_3))
        out_from_e_5 = self.e_block_5(self.conv_5(out_from_e_4))
        out_from_e_6 = self.e_block_6(self.conv_6(out_from_e_5))
        out_from_e = self.conv_7(out_from_e_6)
        #view and fullchain
        out_from_e = out_from_e.view(-1, 640*8*8)
        out_from_fc = self.fc_2(self.fc_1(out_from_e))
        out_from_fc = out_from_fc.view(-1, 640, 8, 8)
        #decording and skip connection
        # input bs x 1280 x 8 x 8
        out_from_de_1 = self.deconv_1(self.de_block_1(out_from_fc + out_from_e_6), output_size=out_from_e_5.size())
        out_from_de_2 = self.deconv_2(self.de_block_2(out_from_de_1 + out_from_e_5), output_size=out_from_e_4.size())
        out_from_de_3 = self.deconv_3(self.de_block_3(out_from_de_2 + out_from_e_4), output_size=out_from_e_3.size())
        out_from_de_4 = self.deconv_4(self.de_block_4(out_from_de_3 + out_from_e_3), output_size=out_from_e_2.size())
        out_from_de_5 = self.deconv_5(self.de_block_5(out_from_de_4 + out_from_e_2), output_size=out_from_e_1.size())
        out_from_de_6 = self.deconv_6(self.de_block_6(out_from_de_5 + out_from_e_1))

        return out_from_de_6

'''Generator at stage-2 (G2)'''
class _netG2(nn.Module):
    def __init__(self):
        super(_netG2, self).__init__()
        #encoder
        #input bs x 6 x 256 x256
        self.conv_1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        # state bs x 64 x 256 x 256
        self.e_block_1 = ResBlock(64)
        self.conv_2 = nn.Conv2d(64, 128, 3, 2, 1)
        #state bs x 128 x 128 x128
        self.e_block_2 = ResBlock(128)
        self.conv_3 = nn.Conv2d(128, 256, 3, 2, 1)
        #state bs x 256 x 64 x64
        self.e_block_3 = ResBlock(256)
        self.conv_4 = nn.Conv2d(256, 384, 3, 2, 1)
        #state bs x 384 x 32 x32
        self.e_block_4 = ResBlock(384)
        #state bs x 384 x 32 x 32

        #decoder
        self.de_block_1 = ResBlock(384)
        self.deconv_1 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1)
        #state bs x 384 x 64 x 64
        self.de_block_2 = ResBlock(256)
        self.deconv_2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        #state bs x 256 x 128 x 128
        self.de_block_3 = ResBlock(128)
        self.deconv_3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        #state bs x 128 x 256 x 256
        self.de_block_4 = ResBlock(64)
        self.deconv_4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)


    def forward(self, input):
        #encoding
        out_from_e_1 = self.e_block_1(self.conv_1(input))
        out_from_e_2 = self.e_block_2(self.conv_2(out_from_e_1))
        out_from_e_3 = self.e_block_3(self.conv_3(out_from_e_2))
        out_from_e_4 = self.e_block_4(self.conv_4(out_from_e_3))
        #decoding
        out_from_de_1 = self.deconv_1(self.de_block_1(out_from_e_4 + out_from_e_4), output_size=out_from_e_3.size())
        out_from_de_2 = self.deconv_2(self.de_block_2(out_from_de_1 + out_from_e_3), output_size=out_from_e_2.size())
        out_from_de_3 = self.deconv_3(self.de_block_3(out_from_de_2 + out_from_e_2), output_size=out_from_e_1.size())
        out_from_de_4 = self.deconv_4(self.de_block_4(out_from_de_3 + out_from_e_1))

        return out_from_de_4

'''Residual Block (-Conv-ReLU-Conv-ReLU-(+shortcut)-)'''
class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        # ch has no change
        self.res_conv_1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.res_relu_1 = nn.ReLU(True)
        self.res_conv_2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.res_relu_2 = nn.ReLU(True)

    def forward(self, x):
        out = self.res_relu_1(self.res_conv_1(x))
        out = self.res_relu_2(self.res_conv_2(out))
        out += x

        return out

'''Discriminator at stage-2 (D)'''
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        ndf = 64
        self.main = nn.Sequential(
            #input bs x 6 x 256 x 256
            nn.Conv2d(6, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state bs x (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state bs x (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state bs x (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state bs x (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            #state bs x (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            #state bs x (ndf) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)

'''custom weights initialization called on netG and netD'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


'''setup network and initialize weights'''
netG1 = _netG1()
netG1.apply(weights_init)
netG2 = _netG2()
netG2.apply(weights_init)
netD = _netD()
netD.apply(weights_init)


'''criterion'''
L1_criterion = nn.L1Loss()
BCE_criterion = nn.BCELoss()


'''define variables'''
condition_Ia = torch.FloatTensor(opt.batchSize, 3, 256, 256)
target_Pb = torch.FloatTensor(opt.batchSize, 18, 256, 256)
target_Ib = torch.FloatTensor(opt.batchSize, 3, 256, 256)
label = torch.FloatTensor(opt.batchSize)

real_label = 1
fake_label = 0

'''using cuda'''
if opt.cuda:
    netG1.cuda()
    netG2.cuda()
    netD.cuda()
    L1_criterion.cuda()
    BCE_criterion.cuda()
    condition_Ia = conditionIa.cuda()
    target_Pb = target_Pb.cuda()
    target_Ib = target_Ib.cuda()
    label = label.cuda()

'''to Variable'''
condition_Ia = Variable(condition_Ia)
target_Pb = Variable(target_Pb)
target_Ib = Variable(target_Ib)
label = Variable(label)

'''setup optimizer'''
optimizerG1 = optim.Adam(netG1.parameters(), lr=2e-5, betas=(0.5, 0.999))
optimizerG2 = optim.Adam(netG2.parameters(), lr=2e-5, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=2e-5, betas=(0.5, 0.999))

'''training G1'''
for epoch in range(opt.niterG1):
    for i, data in enumerate(dataloader, 0):
        netG1.zero_grad()
        condition_Ia_cpu, _ = data[:,0:3,:,:]
        target_Pb_cpu, _ = data[:,3:21,:,:]
        target_Ib_cpu, _ = data[:,21:24,:,:]
        condition_Ia.data.copy_(condition_Ia_cpu)
        target_Pb.data.copy_(target_Pb_cpu)
        target_Ib.data.copy_(target_Ib_cpu)

        input_G1 = torch.cat((condition_Ia, target_Pb), 1) # input_G1 bs x 21 x 256 x256
        pred_Ib = netG1(input_G1)
        errG1 = L1_criterion(pred_Ib, target_Ib) #this is not pose-mask-loss
        errG1.backward()
        optimizerG1.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_G1: %.4f' % (epoch, opt.niterG1, i, len(dataloader), errG1.data[0]))

    if epoch % 10 == 0:
        vutils.save_image(pred_Ib,
            'out/pred_Ib_trainingG1_epoch_%03d.png' % (epoch),
            normalize=True)
    # do checkpointing
    if epoch % 100 == 0:
        torch.save(netG1.state_dict(), 'outpth/netG1_epoch_%d.pth' % (epoch))


'''training Adversarial net (G2 and D)'''
for epoch in range(opt.niterG2):
    for i, data in enumerate(dataloader, 0):
        netG2.zero_grad()
        netD.zero_grad()

        condition_Ia_cpu, _ = data[:,0:3,:,:]
        target_Pb_cpu, _ = data[:,3:21,:,:]
        target_Ib_cpu, _ = data[:,21:24,:,:]
        condition_Ia.data.copy_(condition_Ia_cpu)
        target_Pb.data.copy_(target_Pb_cpu)
        target_Ib.data.copy_(target_Ib_cpu)

        input_G1 = torch.cat((condition_Ia, target_Pb), 1) # input_G1 bs x 21 x 256 x256
        pred_Ib = netG1(input_G1).detach()

        input_G2 = torch.cat((condition_Ia, pred_Ib), 1) # input_G2 bs x 6 x 256 x256
        refined_pred_Ib = netG2(input_G2)

        real_pair = torch.cat((condition_Ia, target_Ib), 1) # input_D bs x 6 x256 x 256
        fake_pair = torch.cat((condition_Ia, refined_pred_Ib), 1) # input_D bs x 6 x256 x 256

        # train D with pairs
        output_real = netD(real_pair)
        label.data.fill_(real_label)
        errD_real = BCE_criterion(output_real, lable)
        errD_real.backward()

        output_fake = netD(fake_pair.detach()) # detach
        lable.data.fill_(fake_label)
        errD_fake = BCE_criterion(output_fake, label)
        errD_fake.backward()

        optimizerD.step()

        # tarin G with pairs
        output_fake = netD(fake_pair)
        label.data.fill_(real_label) # fake labels are real for generator cost
        errG2 = BCE_criterion(output_fake, label)
        errG2 += opt.L1_lambda * L1_criterion(refined_pred_Ib, target_Ib)
        errG2.backward()

        optimizerG2.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_G2: %.4f' % (epoch, opt.niterG2, i, len(dataloader), errG2.data[0]))
    
    if epoch % 10 == 0:
        vutils.save_image(refined_pred_Ib,
            'out/refined_pred_Ib_trainingG2_epoch_%03d.png' % (epoch),
            normalize=True)

    # do checkpointing
    if epoch % 100 == 0:
        torch.save(netG2.state_dict(), 'outpth/netG2_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), 'outpth/netD_epoch_%d.pth' % (epoch))
