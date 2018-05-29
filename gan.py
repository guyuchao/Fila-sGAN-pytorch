from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
class UpBlock(nn.Module):
    def __init__(self,inplanes,outplanes,Relu='L',padding=1):
        super(UpBlock,self).__init__()
        if Relu=='L':
            self.relu=nn.LeakyReLU(0.2)
        elif Relu=='R':
            self.relu=nn.ReLU()
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(inplanes,outplanes,kernel_size=4,stride=2,padding=padding),
            nn.BatchNorm2d(outplanes),

        )
    def forward(self, x):
        x=self.relu(x)
        x=self.decoder(x)
        return x

class Block(nn.Module):
    def __init__(self,inplanes,outplanes,Relu='R',padding=1):
        super(Block,self).__init__()
        if Relu=='R':
            self.relu=nn.ReLU()
        elif Relu=='L':
            self.relu=nn.LeakyReLU(0.2)
        self.encoder=nn.Sequential(
            nn.Conv2d(inplanes,outplanes,kernel_size=4,stride=2,padding=padding),
            nn.BatchNorm2d(outplanes),
        )

    def forward(self, x):
        x=self.encoder(x)
        x=self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self,filters=64):
        super(Generator, self).__init__()
        self.filters=filters
        self.layer0=nn.Sequential(
            nn.Conv2d(2,filters,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2))
        self.layer1 = Block(filters,2*filters)
        self.layer2 = Block(2*filters, 4 * filters)
        self.layer3 = Block(4*filters, 8 * filters)
        self.layer4 = Block(8*filters, 8 * filters,padding=0)
        self.layer5 = Block(8*filters, 8 * filters)

        self.z1=nn.Sequential(
            nn.Linear(400,4*4*filters),
            nn.LeakyReLU(0.2))
        self.gen1=UpBlock(filters,2*filters,Relu='L')
        self.gen2=UpBlock(2*filters+8*filters,8*filters,Relu='L')
        self.gen3 = UpBlock(8 * filters + 8 * filters, 8 * filters,Relu='R',padding=0)
        self.gen4 = UpBlock(8 * filters + 8 * filters, 4 * filters,Relu='R')
        self.gen5 = UpBlock(8 * filters, 2 * filters,Relu='R')

        self.gen6 = UpBlock(4 * filters,  filters,Relu='R')

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(filters*2,3,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0,0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0,0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x,z):
        #print(x.size())
        x0 = self.layer0(x)
        #print(x0.size())
        x1 = self.layer1(x0)
        #print(x1.size())
        x2 = self.layer2(x1)
        #print(x2.size())
        x3 = self.layer3(x2)
        #print(x3.size())
        x4 = self.layer4(x3)
        #print(x4.size())
        x5 = self.layer5(x4)
        #print(x5.size())
        z1=self.z1(z)
        z1=z1.view((-1,self.filters,4,4))

        z2=self.gen1(z1)
        #print(z2.size(),x5.size())


        z2=torch.cat((z2,x5),dim=1)

        z3=self.gen2(z2)
        #print(z3.size(),x4.size())
        z3 = torch.cat((z3, x4), dim=1)

        z4 = self.gen3(z3)
        z4=F.pad(z4,(1,0,1,0))

        #print(z4.size(),x3.size())
        z4 = torch.cat((z4, x3), dim=1)

        z5=self.gen4(z4)
        #print(z5.size(),x2.size())
        #z5=F.pad(z5,(1,0,1,0))

        z5=torch.cat((z5, x2), dim=1)

        z6 = self.gen5(z5)
        #print(z6.size(), x1.size())
        z6 = F.pad(z6, (1, 0, 1, 0))
        z6 = torch.cat((z6, x1), dim=1)

        z7 = self.gen6(z6)
        z7 = F. torch.cat((z7, x0), dim=1)

        out=self.out(z7)

        return out


class Discriminator(nn.Module):
    def __init__(self,filters=32):
        super(Discriminator, self).__init__()
        self.filters=filters
        self.layer0 = nn.Sequential(
            nn.Conv2d(5, filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
        self.layer1 = Block(filters,2*filters,Relu='L')
        self.layer2 = Block(2*filters, 4 * filters, Relu='L')
        self.layer3 = Block(4*filters, 8 * filters, Relu='L')
        self.layer4 = Block(8*filters, 16 * filters, Relu='L')
        self.out_digit=nn.Linear(16*filters*17*17,1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x=self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=x.view(x.size()[0],-1)
        out=self.out_digit(x)
        return F.sigmoid(out)

if __name__=='__main__':
    a=Variable(torch.ones((2,2,564,564))).cuda()
    b=Variable(torch.randn((2,400))).cuda()
    model=Generator().cuda()
    print(model(a,b).size())
