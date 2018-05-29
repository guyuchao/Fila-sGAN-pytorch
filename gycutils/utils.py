import torch
from torch.autograd import Variable,grad
def make_trainable(model, val):
    for p in model.parameters():
        p.requires_grad = val


def calc_gradient_penalty(netD, real_data, fake_data,LAMBDA=10):
    BATCH=real_data.size()[0]
    alpha = torch.rand(BATCH, 1)
    #print(alpha.size(),real_data.size())
    alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand(real_data.size())
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def Gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G



def get_content_features(Vgg_net,img,mask):
    img=(img+1)*127.5
    img=img*((mask+1)/2)
    content_features = Vgg_net(img)[1]
    return content_features

def get_style_features(Vgg_net,img,mask):
    img=(img+1)*127.5
    img=img*((mask+1)/2)
    style_features = Vgg_net(img)
    # style_gram = [gram(fmap) for fmap in style_features]
    #style_feature_x = {}
    #style_feature_y = {}
    style_feature = {}
    for idx, feature in enumerate(style_features):
        #feature_x = feature[:, :, 1:, :] - feature[:, :, :-1, :]
        #feature_y = feature[:, :, :, 1:] - feature[:, :, :, :-1]
        #gram_x = Gram(feature_x)
        #gram_y = Gram(feature_y)
        gram = Gram(feature)
        #style_feature_x[idx] = gram_x
        #style_feature_y[idx] = gram_y
        style_feature[idx] = gram
    return style_feature

def get_style_loss(style_feature,fake_style_feature):
    style_loss=0.0
    for i in range(4):
        coff=float(1.0/4)
        fake_gram=fake_style_feature[i]
        style_gram=style_feature[i]
        style_loss+=coff*torch.mean(torch.abs((fake_gram-style_gram)))
    style_loss=torch.mean(style_loss)
    return style_loss

def get_content_loss(content_feature_real,content_feature_fake):
    coff=1
    content_loss=coff*torch.mean(torch.abs(content_feature_fake-content_feature_real))
    return content_loss

def get_tv_loss(img):
    x = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    y = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return x+y
