from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
from Vgg import Vgg16
from gycutils.trainschedule import Scheduler
from gycutils.utils import make_trainable,get_tv_loss,get_content_loss,get_content_features,get_style_features,get_style_loss
from gan import Discriminator,Generator
from datasets import VOCDataSet
from torch.optim import Adam
from loss import BCE_Loss
from transform import ReLabel, ToLabel
from torchvision.transforms import Compose, Normalize, ToTensor,Resize
import tqdm
from Criterion import Criterion
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
from gycutils.gycaug import Img_to_zero_center,Random_horizontal_flip,Random_vertical_flip,Compose_imglabel,Random_crop
input_transform = Compose([
    #Resize(512, interpolation=Image.BICUBIC),
    ToTensor(),
    Img_to_zero_center()
    ])
target_transform = Compose([
    #Resize(512, interpolation=Image.BICUBIC),

    ToLabel(),
    ReLabel(255, 1),
    Img_to_zero_center()
    ])
mask_transform = Compose([
    #Resize(512, interpolation=Image.BICUBIC),
    ToLabel(),
    ReLabel(255, 1),
    Img_to_zero_center()
    ])

pth="./eyedata/style/style.tif"
#style_imgs=[]
#for filename in os.listdir(pth):
 #   if len(style_imgs)>5:
  #      break
style_img=Image.open(pth)
style_img=Variable(input_transform(style_img),requires_grad=False).cuda()
trainloader = data.DataLoader(VOCDataSet("./", img_transform=input_transform,
                                         label_transform=target_transform,mask_transform=mask_transform),
                              batch_size=1, shuffle=True, pin_memory=True)

valloader = data.DataLoader(VOCDataSet("./", split='val',img_transform=input_transform,
                                         label_transform=target_transform,mask_transform=mask_transform),
                              batch_size=1, shuffle=True, pin_memory=True)

#########################################
#Parameters
#adversarial
L_gan_weight=1
#style
L_style_weight=10
#content
L_content_weight=1
#tv
L_tv_weight=100

lr=0.0002
beta1=0.5
batch_size=1
max_epoch=2000

channel=3
img_size=564
img_x=564
img_y=564

style_size=564
sample_batch=4
z_size=400
#########################################
def gen_rand_noise(batch_size,z_size,mean=0,std=0.001):
    z_sample = np.random.normal(mean, std, size=[batch_size, z_size]).astype(np.float32)
    z=torch.from_numpy(z_sample)
    return z
#########################################
G=Generator().cuda()
D=Discriminator().cuda()
Vgg=Vgg16().cuda()
bce=BCE_Loss()
mse=torch.nn.MSELoss()
optimizer_d = torch.optim.SGD(D.parameters(), lr=lr*0.4)
optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.9))
#########################################
#########################################
for epoch in range(max_epoch):
    for idx, (real_img, real_label,mask) in tqdm.tqdm(enumerate(trainloader)):
        # trainD
        make_trainable(D, True)
        make_trainable(G, False)

        D.zero_grad()
        optimizer_d.zero_grad()

        real_img = Variable(real_img).cuda()
        real_label = Variable(real_label.unsqueeze(1)).cuda()
        mask=Variable(mask.unsqueeze(1)).cuda()

        z = Variable(gen_rand_noise(batch_size,z_size)).cuda()
        real_label_mask_pair=torch.cat((real_label,mask),dim=1)
        fake_imgs=G(real_label_mask_pair,z)

        real_pair=torch.cat((real_img,real_label,mask),dim=1)
        fake_pair=torch.cat((fake_imgs,real_label,mask),dim=1)

        D_real_logits=D(real_pair)
        D_real_y=Variable(torch.ones((batch_size,1))).cuda()

        D_fake_logits=D(fake_pair)
        D_fake_y=Variable(torch.zeros((batch_size,1))).cuda()

        d_real_loss=bce(D_real_logits,D_real_y)
        d_fake_loss=bce(D_fake_logits,D_fake_y)

        d_loss=d_real_loss+d_fake_loss
        d_loss.backward()
        optimizer_d.step()

        #trainG twice
        make_trainable(G, True)
        make_trainable(D, False)
        make_trainable(Vgg,False)
        for _ in range(1):
            G.zero_grad()
            optimizer_g.zero_grad()

            z = Variable(gen_rand_noise(batch_size, z_size,0,0.5)).cuda()
            real_label_mask_pair = torch.cat((real_label, mask), dim=1)
            fake_imgs = G(real_label_mask_pair, z)
            fake_pair=torch.cat((fake_imgs,real_label,mask),dim=1)

            D_fake_logits=D(fake_pair)
            D_fake_y=Variable(torch.ones((batch_size,1))).cuda()
            #gan_loss
            g_loss_adversial=bce(D_fake_logits,D_fake_y)
            #style loss
            #style_loss=0.0
            #for i,style_img in enumerate(style_imgs):
            style_feature = get_style_features(Vgg, style_img,mask)
            style_loss=get_style_loss(style_feature,get_style_features(Vgg,fake_imgs,mask))#/len(style_imgs)
            #content_loss
            content_loss=get_content_loss(get_content_features(Vgg,real_img,mask),get_content_features(Vgg,fake_imgs,mask))
            #tv_loss
            tv_loss=get_tv_loss(fake_imgs)
            loss=L_gan_weight*g_loss_adversial+L_style_weight*style_loss+L_content_weight*content_loss+L_tv_weight*tv_loss
            loss.backward()
            optimizer_g.step()

    print("epoch[%d/%d] d_loss:%.4f g_loss_ad:%.4f style_loss:%.4f content_loss:%.4f tv_loss%.4f"%(
        epoch,max_epoch,d_loss,g_loss_adversial,style_loss,content_loss,tv_loss
    ))
    if epoch%50==0:
        G.eval()
        D.eval()

        os.mkdir('./pth/epoch%d'%epoch)
        for idx, (real_img, real_label, mask) in tqdm.tqdm(enumerate(valloader)):
            os.mkdir('./pth/epoch%d/label%d' %(epoch,idx) )
            real_img = Variable(real_img).cuda()
            real_label = Variable(real_label.unsqueeze(1)).cuda()
            img_label=real_label.squeeze(1).cpu().data[0].numpy()
            #print(img_label.shape)
            Image.fromarray(img_label.astype(np.uint8)).save('./pth/epoch%d/label%d/label.jpg'%(epoch,idx))
            mask = Variable(mask.unsqueeze(1)).cuda()

            for i in range(5):
                z=Variable(gen_rand_noise(1,z_size,0,0.5)).cuda()
                real_label_mask_pair = torch.cat((real_label, mask), dim=1)
                fake_imgs = G(real_label_mask_pair, z)[0].cpu().data.numpy()
                img=np.transpose(fake_imgs,[1,2,0])

                img=(img+1)*127.5
                img=Image.fromarray(img.astype(np.uint8))
                img.save('./pth/epoch%d/label%d/%d.jpg'%(epoch,idx,i))


