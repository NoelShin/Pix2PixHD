import torch
import torch.nn as nn
import torchvision
from utils import get_norm_layer, get_pad_layer, get_grid


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.set_weight()
        self.act = nn.ReLU(inplace=True)
        self.max_gf = opt.n_gf * 2**opt.n_downsample
        self.norm = get_norm_layer(opt.norm_type)
        self.pad = get_pad_layer(opt.padding_type)
        print(self)

    def set_weight(self):
        self.first_conv = nn.Conv2d(self.opt.input_ch, self.opt.n_gf, kernel_size=7, padding=0)
        self.last_conv = nn.Conv2d(self.opt.n_gf, self.opt.output_ch, kernel_size=7, padding=0)
        self.enc_list = nn.ModuleList([nn.Conv2d(self.opt.n_gf * 2 ** i, self.opt.n_gf * 2 ** (i + 1),
                                                 kernel_size=3, padding=1, stride=2)
                                       for i in range(self.opt.n_downsample)])
        self.res_list = nn.ModuleList([nn.Conv2d(self.opt.n_gf * 2 ** self.opt.n_downsample,
                                                 self.opt.n_gf * 2 ** self.opt.n_downsample,
                                                 kernel_size=3, padding=0, stride=1)
                                       for i in range(self.opt.n_residual * 2)])
        self.dec_list = nn.ModuleList([nn.ConvTranspose2d(self.opt.n_gf * 2 ** (i + 1), self.opt.n_gf * 2 ** i,
                                                          kernel_size=3, padding=1, stride=2, output_padding=1)
                                       for i in range(self.opt.n_downsample - 1, -1, -1)])

    def encoder(self, x):
        model = []
        n_gf = self.opt.n_gf
        model += [self.pad(3), self.first_conv, self.norm(n_gf), self.act]

        for i in range(self.opt.n_downsample):
            n_gf *= 2
            model += [self.enc_list[i], self.norm(n_gf), self.act]

        x = nn.Sequential(*model)(x)

        return x

    def residual_block(self, x):
        for i in range(self.opt.n_residual):
            y = self.act(self.norm(self.max_gf)(self.res_list[2*i](self.pad(1)(x))))
            y = self.norm(self.max_gf)(self.res_list[2*i + 1](self.pad(1)(y)))
            x = x + y

        return x

    def decoder(self, x):
        model = []
        n_gf = self.opt.n_gf * 2**self.opt.n_downsample

        for i in range(self.opt.n_downsample):
            n_gf /= 2
            model += [self.dec_list[i], self.norm(int(n_gf)), self.act]

        model += [self.pad(3), self.last_conv, nn.Tanh()]

        x = nn.Sequential(*model)(x)

        return x

    def forward(self, x):
        return self.decoder(self.residual_block(self.encoder(x)))


class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        self.opt = opt

        self.norm = get_norm_layer(opt.norm_type)
        self.set_weight()
        self.build_model()

    def set_weight(self):
        input_channel = self.opt.input_ch + self.opt.output_ch
        n_df = self.opt.n_df
        self.conv_1 = nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2)  # 1x64x128x128
        self.conv_2 = nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2)  # 1x128x64x64
        self.conv_3 = nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2)  # 1x256x32x32
        self.conv_4 = nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1)  # 1x512x31x31
        self.conv_5 = nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)  # 1x512x30x30

    def build_model(self):
        n_df = self.opt.n_df
        blocks = []
        blocks += [[self.conv_1, nn.LeakyReLU(0.2, inplace=True)]]
        blocks += [[self.conv_2, self.norm(2 * n_df), nn.LeakyReLU(0.2, inplace=True)]]
        blocks += [[self.conv_3, self.norm(4 * n_df), nn.LeakyReLU(0.2, inplace=True)]]
        blocks += [[self.conv_4, self.norm(8 * n_df), nn.LeakyReLU(0.2, inplace=True)]]
        blocks += [[self.conv_5]] if not self.opt.GAN_type == 'GAN' else [[self.conv_5, nn.Sigmoid()]]
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
            fake_grid = get_grid(fake_features[i][-1], is_real=False)  # it doesn't need to be fake_score

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
            real_features, fake_features = self.VGGNet(fake), self.VGGNet(target)

            for i in range(len(real_features)):
                loss_G_VGG_FM += weights[i] * self.FMcriterion(real_features[i].detach(), fake_features[i])

            loss_G += loss_G_VGG_FM * self.opt.lambda_FM

        return loss_D, loss_G, target, fake
