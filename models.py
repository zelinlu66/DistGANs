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
from torch import nn

F = torch.nn.functional


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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Res net models: Discriminator
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                outchannel,
                outchannel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
        )
        self.skip = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    inchannel,
                    self.expansion * outchannel,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * outchannel),
            )

    def forward(self, X):
        out = F.relu(self.conv1(X))
        out = self.conv2(out)
        out += self.skip(X)
        out = F.relu(out)
        return out


class ResNetD(nn.Module):
    def __init__(self, ResidualBlock, num_classes):
        super(ResNetD, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512 * ResidualBlock.expansion, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def to(self, device):
        super(ResNetD, self).to(device)
        self.device = device


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# ResNet model: Generator


class ResidualBlock_(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = self.conv_bn_relu(ch)
        self.conv2 = self.conv_bn_relu(ch)

    def conv_bn_relu(self, ch):
        return nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        x = self.conv2(self.conv1(inputs))
        return inputs + x


class ResNetG(nn.Module):
    def __init__(self, upsampling_type):
        assert upsampling_type in [
            "nearest_neighbor",
            "transpose_conv",
            "pixel_shuffler",
        ]
        self.upsampling_type = upsampling_type
        super().__init__()
        self.inital = nn.Sequential(
            nn.Conv2d(100, 768, 1), nn.BatchNorm2d(768), nn.ReLU(True)
        )
        self.conv1 = self.generator_block(768, 512, 7, 2)
        self.conv2 = self.generator_block(512, 256, 2, 2)
        self.conv3 = self.generator_block(256, 128, 2, 2)
        self.conv4 = self.generator_block(128, 64, 2, 2)
        self.conv5 = self.generator_block(64, 32, 2, 2)
        self.conv6 = self.generator_block(32, 16, 2, 1)
        self.out = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1), nn.Tanh()
        )

    def generator_block(
        self, in_ch, out_ch, upsampling_factor, n_residual_block
    ):
        layers = []
        if self.upsampling_type == "transpose_conv":
            layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=upsampling_factor,
                    stride=upsampling_factor,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(True))
        elif self.upsampling_type == "nearest_neighbor":
            layers.append(
                nn.UpsamplingNearest2d(scale_factor=upsampling_factor)
            )
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(True))
        elif self.upsampling_type == "pixel_shuffler":
            layers.append(
                nn.Conv2d(
                    in_ch, out_ch * upsampling_factor ** 2, kernel_size=1
                )
            )
            layers.append(nn.BatchNorm2d(out_ch * upsampling_factor ** 2))
            layers.append(nn.ReLU(True))
            layers.append(nn.PixelShuffle(upscale_factor=upsampling_factor))

        for i in range(n_residual_block):
            layers.append(ResidualBlock_(out_ch))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.conv6(
            self.conv5(
                self.conv4(
                    self.conv3(self.conv2(self.conv1(self.inital(inputs))))
                )
            )
        )
        return self.out(x)

    def to(self, device):
        super(ResNetG, self).to(device)
        self.device = device
