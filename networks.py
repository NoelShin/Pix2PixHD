import torch
import torch.nn as nn
import torchvision
from utils import get_norm_layer, get_pad_layer, get_grid


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.build_model()
        print(self)

    def build_model(self):
        act = nn.ReLU(inplace=True)
        n_gf = self.opt.n_gf
        norm = get_norm_layer(self.opt.norm_type)
        pad = get_pad_layer(self.opt.padding_type)

        model = []
        model += [pad(3), nn.Conv2d(self.opt.input_ch, n_gf, kernel_size=7, padding=0), act]

        for _ in range(self.opt.n_downsample):
            model += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=1, stride=2), norm(2 * n_gf), act]
            n_gf *= 2

        for _ in range(self.opt.n_residual):
            model += [ResidualBlock(n_gf, pad, norm, act)]

        for _ in range(self.opt.n_downsample):
            model += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                      norm(n_gf//2), act]
            n_gf //= 2

        model += [pad(3), nn.Conv2d(n_gf, self.opt.output_ch, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        self.opt = opt
        self.build_model()

    def build_model(self):
        input_channel = self.opt.input_ch + self.opt.output_ch
        n_df = self.opt.n_df
        norm = get_norm_layer(self.opt.norm_type)
        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2),
                    nn.LeakyReLU(0.2, inplace=True)]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2),
                    norm(2 * n_df), nn.LeakyReLU(0.2, inplace=True)]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2),
                    norm(4 * n_df), nn.LeakyReLU(0.2, inplace=True)]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1),
                    norm(8 * n_df), nn.LeakyReLU(0.2, inplace=True)]]

        if not self.opt.GAN_type == 'GAN':
            blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        else:
            blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1), nn.Sigmoid()]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))
        return result[1:]  # except for the input


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt

        if opt.GAN_type == 'GAN':
            pass

        elif opt.GAN_type == 'LSGAN':
            for i in range(opt.n_D):
                setattr(self, 'Scale_{}'.format(i), PatchDiscriminator(self.opt))

        elif opt.GAN_type == 'WGAN_GP':
            pass

        print(self)

    def forward(self, x):
        result = []
        for i in range(self.opt.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)(x)

        return result


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg_pretrained_layers = torchvision.models.vgg19(pretrained=True).features
        self.block_1 = nn.Sequential()
        self.block_2 = nn.Sequential()
        self.block_3 = nn.Sequential()
        self.block_4 = nn.Sequential()
        self.block_5 = nn.Sequential()

        for i in range(2):
            self.block_1.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(2, 7):
            self.block_2.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(7, 12):
            self.block_3.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(12, 21):
            self.block_4.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(21, 30):
            self.block_5.add_module(str(i), vgg_pretrained_layers[i])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_2(out_1)
        out_3 = self.block_3(out_2)
        out_4 = self.block_4(out_3)
        out_5 = self.block_5(out_4)
        out = [out_1, out_2, out_3, out_4, out_5]

        return out


class Loss(object):
    def __init__(self, opt):
        self.opt = opt
        if opt.GAN_type == 'GAN':
            self.criterion = nn.BCELoss()

        elif opt.GAN_type == 'LSGAN':
            self.criterion = nn.MSELoss()
            self.FMcriterion = nn.L1Loss()

            if opt.VGG_loss:
                self.VGGNet = VGG19()

                if opt.gpu_ids != -1:
                    self.VGGNet = self.VGGNet.cuda(opt.gpu_ids)

    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake = G(input)
        fake_features = D(torch.cat([input, fake.detach()], dim=1))
        real_features = D(torch.cat([input, target], dim=1))

        for i in range(self.opt.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True)
            fake_grid = get_grid(fake_features[i][-1], is_real=False)  # it doesn't need to be fake_features

            if self.opt.gpu_ids != -1:
                real_grid = real_grid.cuda(self.opt.gpu_ids)
                fake_grid = fake_grid.cuda(self.opt.gpu_ids)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        fake_features = D(torch.cat([input, fake], dim=1))

        for i in range(self.opt.n_D):
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())

            real_grid = get_grid(fake_features[i][-1], is_real=True)
            if self.opt.gpu_ids != -1:
                real_grid = real_grid.cuda(self.opt.gpu_ids)
            loss_G += self.criterion(fake_features[i][-1], real_grid)

        loss_G += loss_G_FM * (1.0/self.opt.n_D) * self.opt.lambda_FM

        if self.opt.VGG_loss:
            loss_G_VGG_FM = 0
            weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            real_features, fake_features = self.VGGNet(target), self.VGGNet(fake)

            for i in range(len(real_features)):
                loss_G_VGG_FM += weights[i] * self.FMcriterion(fake_features[i], real_features[i])
            loss_G += loss_G_VGG_FM * self.opt.lambda_FM

        return loss_D, loss_G, target, fake


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        self.build_block(n_channels, pad, norm, act)

    def build_block(self, n_channels, pad, norm, act):
        block = [pad(1)]
        block += [nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1)]
        block += [nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = x + self.block(x)

        return x
