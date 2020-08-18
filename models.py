# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:26:08 2020

@authors: Andrey Prokpenko (e-mail: prokopenkoav@ornl.gov)
        : Debangshu Mukherjee (e-mail: mukherjeed@ornl.gov)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
        : Vitaliy Starchenko  (e-mail: starchenkov@ornl.gov)
        : Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it) 

"""
import torch
import numpy
from ResNet_utils import Lambda, NormalizeLayer, Conv2dEx, LinearEx, ResBlock2d
from torch import nn
from abc import ABC, abstractmethod


class Discriminator_MLP(nn.Module):
    def __init__(self, n_features):
        super(Discriminator_MLP, self).__init__()
        self.hidden0 = nn.Sequential(nn.Linear(n_features, 250), nn.ReLU())
        self.hidden1 = nn.Sequential(nn.Linear(250, 100), nn.ReLU())
        # self.out = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(100, 1))

    def forward(self, input):
        x = self.hidden0(input)
        y = self.hidden1(x)
        z = self.out(y)
        return z

    def to(self, device):
        super(Discriminator_MLP, self).to(device)
        self.device = device


class Generator_MLP(torch.nn.Module):
    def __init__(self, noise_dimension, n_out):
        super(Generator_MLP, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(noise_dimension, 1000), nn.ReLU()
        )  #  ORIGINAL:   nn.LeakyReLU(0.2))
        self.hidden1 = nn.Sequential(nn.Linear(1000, 1000), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(1000, 1000), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(1000, n_out), nn.Tanh())

    def forward(self, input):
        x = self.hidden0(input)
        y = self.hidden1(x)
        w = self.hidden2(y)
        z = self.out(w)
        return z

    def to(self, device):
        super(Generator_MLP, self).to(device)
        self.device = device


class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor
        )


class GeneratorCNN(nn.Module):
    def __init__(self, noise_dimension, n_channels, image_dimension):
        super(GeneratorCNN, self).__init__()

        self.init_size = image_dimension // 4
        self.l1 = nn.Sequential(
            nn.Linear(noise_dimension, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            Upsample(scale_factor=2),  # nn.Upsample
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, n_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def to(self, device):
        super(GeneratorCNN, self).to(device)
        self.device = device


class DiscriminatorCNN(nn.Module):
    def __init__(self, n_channels, image_dimension):
        super(DiscriminatorCNN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(n_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = image_dimension // 2 ** 4
        self.adv_layer = nn.Sequential(
            # nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid()
            nn.Linear(128 * ds_size ** 2, 1)
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

    def to(self, device):
        super(DiscriminatorCNN, self).to(device)
        self.device = device


##########################


class ConditionalGenerator_MLP(nn.Module):
    def __init__(self, img_shape, n_classes, latent_dim=100):
        super(ConditionalGenerator_MLP, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(numpy.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def to(self, device):
        super(ConditionalGenerator_MLP, self).to(device)
        self.device = device


class ConditionalDiscriminator_MLP(nn.Module):
    def __init__(self, img_shape, n_classes):
        super(ConditionalDiscriminator_MLP, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(numpy.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat(
            (img.view(img.size(0), -1), self.label_embedding(labels)), -1
        )
        validity = self.model(d_in)
        return validity

    def to(self, device):
        super(ConditionalDiscriminator_MLP, self).to(device)
        self.device = device


class ConditionalGenerator_CNN(nn.Module):
    def __init__(self, img_shape, n_classes, latent_dim=100):
        super(ConditionalGenerator_CNN, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.img_shape = img_shape
        self.init_size = self.img_shape[1] // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            Upsample(scale_factor=2),  # nn.Upsample
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def to(self, device):
        super(ConditionalGenerator_CNN, self).to(device)
        self.device = device


class ConditionalDiscriminator_CNN(nn.Module):
    def __init__(self, img_shape, n_classes):
        super(ConditionalDiscriminator_CNN, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.img_shape = img_shape
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(img_shape[0], 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(128, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128 * 2, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf*4) x 8 x 8
        # nn.Conv2d(128 * 4, 128 * 8, 4, 2, 1, bias=False),
        # nn.BatchNorm2d(128 * 8),
        # nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        # nn.Conv2d(128 * 8, 1, 4, 1, 0, bias=False),
        # nn.Sigmoid()
        self.second = nn.Sequential(
            nn.Linear(n_classes, 1000), nn.LeakyReLU(0.2, inplace=True)
        )
        self.third = nn.Sequential(
            nn.Linear(1000 + 128 * 4 * 8 * 8, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        x = self.main(img)
        y = self.second(self.label_embedding(labels))
        x = x.view(img.size(0), 128 * 4 * 8 * 8)  # 128*np.prod(self.img_shape)
        x = torch.cat([x, y], 1)
        out = self.third(x)

        return out

    def to(self, device):
        super(ConditionalDiscriminator_CNN, self).to(device)
        self.device = device


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Res net models

FMAP_G = 64
FMAP_D = 64
FMAP_G_INIT_64_FCTR = 4
RES_FEATURE_SPACE = 4
FMAP_SAMPLES = 3
RES_INIT = 4


class GAN(nn.Module, ABC):
    """Base GANs architechture for ResNet """

    def __init__(self, res):
        super(GAN, self).__init__()
        self._res = res

    def most_parameters(self, recurse=True, excluded_params: list = []):
        """torch.nn.Module.parameters() generator method but with the option to exclude specified parameters."""
        for name, params in self.named_parameters(recurse=recurse):
            if name not in excluded_params:
                yield params

    @property
    def res(self):
        return self._res

    @res.setter
    def res(self, new_res):
        message = f'GAN().res cannot be changed, as {self.__class__.__name__} only permits one resolution: {self._res}.'
        raise AttributeError(message)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError(
            'Can only call `forward` on valid subclasses.'
        )


class GeneratorResnet(GAN):
    def __init__(
        self,
        len_latent=128,
        fmap=FMAP_G,
        upsampler=nn.Upsample(scale_factor=2, mode='nearest'),
        blur_type=None,
        nl=nn.ReLU(),
        num_classes=0,
        equalized_lr=False,
        FMAP_SAMPLES=3,
    ):
        super(GeneratorResnet, self).__init__(64)

        self.len_latent = len_latent
        self.num_classes = num_classes

        self.equalized_lr = equalized_lr

        _fmap_init_64 = len_latent * FMAP_G_INIT_64_FCTR
        self.generator_model = nn.Sequential(
            Lambda(lambda x: x.view(-1, len_latent + num_classes)),
            LinearEx(
                nin_feat=len_latent + num_classes,
                nout_feat=_fmap_init_64 * RES_INIT ** 2,
                init='Xavier',
                equalized_lr=equalized_lr,
            ),
            Lambda(lambda x: x.view(-1, _fmap_init_64, RES_INIT, RES_INIT)),
            ResBlock2d(
                ni=_fmap_init_64,
                nf=8 * fmap,
                ks=3,
                norm_type='BatchNorm',
                upsampler=upsampler,
                init='He',
                nl=nl,
                equalized_lr=equalized_lr,
                blur_type=blur_type,
            ),
            ResBlock2d(
                ni=8 * fmap,
                nf=4 * fmap,
                ks=3,
                norm_type='BatchNorm',
                upsampler=upsampler,
                init='He',
                nl=nl,
                equalized_lr=equalized_lr,
                blur_type=blur_type,
            ),
            ResBlock2d(
                ni=4 * fmap,
                nf=2 * fmap,
                ks=3,
                norm_type='BatchNorm',
                upsampler=upsampler,
                init='He',
                nl=nl,
                equalized_lr=equalized_lr,
                blur_type=blur_type,
            ),
            ResBlock2d(
                ni=2 * fmap,
                nf=1 * fmap,
                ks=3,
                norm_type='BatchNorm',
                upsampler=upsampler,
                init='He',
                nl=nl,
                equalized_lr=equalized_lr,
                blur_type=blur_type,
            ),
            NormalizeLayer('BatchNorm', ni=1 * fmap),
            nl,
            Conv2dEx(
                ni=1 * fmap,
                nf=FMAP_SAMPLES,
                ks=3,
                stride=1,
                padding=1,
                init='He',
                equalized_lr=equalized_lr,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator_model(x)

    def to(self, device):
        super(GeneratorResnet, self).to(device)
        self.device = device


class DiscriminatorResnet(GAN):
    def __init__(
        self,
        fmap=FMAP_D,
        pooler=nn.AvgPool2d(kernel_size=2, stride=2),
        blur_type=None,
        nl=nn.ReLU(),
        num_classes=0,
        equalized_lr=False,
        FMAP_SAMPLES=3,
    ):
        super(DiscriminatorResnet, self).__init__(64)

        self.num_classes = num_classes
        self.equalized_lr = equalized_lr

        self.view1 = Lambda(
            lambda x: x.view(
                -1, FMAP_SAMPLES + num_classes, self.res, self.res
            )
        )
        self.conv1 = Conv2dEx(
            ni=FMAP_SAMPLES + num_classes,
            nf=1 * fmap,
            ks=3,
            stride=1,
            padding=1,
            init='Xavier',
            equalized_lr=equalized_lr,
        )
        self.resblocks = nn.Sequential(
            ResBlock2d(
                ni=1 * fmap,
                nf=2 * fmap,
                ks=3,
                norm_type='LayerNorm',
                pooler=pooler,
                init='He',
                nl=nl,
                res=self.res // 1,
                equalized_lr=equalized_lr,
                blur_type=blur_type,
            ),
            ResBlock2d(
                ni=2 * fmap,
                nf=4 * fmap,
                ks=3,
                norm_type='LayerNorm',
                pooler=pooler,
                init='He',
                nl=nl,
                res=self.res // 2,
                equalized_lr=equalized_lr,
                blur_type=blur_type,
            ),
            ResBlock2d(
                ni=4 * fmap,
                nf=8 * fmap,
                ks=3,
                norm_type='LayerNorm',
                pooler=pooler,
                init='He',
                nl=nl,
                res=self.res // 4,
                equalized_lr=equalized_lr,
                blur_type=blur_type,
            ),
            ResBlock2d(
                ni=8 * fmap,
                nf=8 * fmap,
                ks=3,
                norm_type='LayerNorm',
                pooler=pooler,
                init='He',
                nl=nl,
                res=self.res // 8,
                equalized_lr=equalized_lr,
                blur_type=blur_type,
            ),
            Lambda(
                lambda x: x.view(-1, RES_FEATURE_SPACE ** 2 * 8 * fmap)
            )  # final feature space
            # NormalizeLayer( 'LayerNorm', ni = fmap, res = 1 )
        )
        self.linear1 = LinearEx(
            nin_feat=RES_FEATURE_SPACE ** 2 * 8 * fmap,
            nout_feat=1,
            init='Xavier',
            equalized_lr=equalized_lr,
        )

    def forward(self, x):
        return (self.linear1(self.resblocks(self.conv1(self.view1(x))))).view(
            -1
        )

    def to(self, device):
        super(DiscriminatorResnet, self).to(device)
        self.device = device
