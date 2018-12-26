import os
from functools import partial

import torch
import torch.nn as nn

import numpy as np

from PIL import Image

def configure(opt):
    torch.cuda.device(opt.gpu_ids)

    opt.n_df = 64

    if opt.dataset_name == 'Cityscapes':
        if opt.use_boundary_map:
            opt.input_ch = 36

        else:
            opt.input_ch = 35

        opt.format = 'png'
        opt.half = True
        opt.image_size = (512, 1024) if opt.half else (1024, 2048)
        opt.n_gf = 64 if opt.half else 32
        opt.output_ch = 3

    else:
        raise NotImplementedError("Please check your image_height. It should be in [512, 1024].")

    dataset_name = opt.dataset_name
    model_name = model_namer(height=opt.image_height)
    make_dir(dataset_name, model_name, type='checkpoints')
    log_path = os.path.join('./checkpoints/', dataset_name, 'Model', model_name + '_opt.txt')

    if os.path.isfile(log_path):
        permission = input(
            "{} log already exists. Do you really want to overwrite this log? Y/N. : ".format(model_name + '_opt'))
        if permission == 'Y':
            pass
        else:
            raise NotImplementedError("Please check {}".format(log_path))

    args = vars(opt)
    with open(log_path, 'wt') as log:
        log.write('-' * 50 + 'Options' + '-' * 50 + '\n')
        print('-' * 50 + 'Options' + '-' * 50)
        for k, v in sorted(args.items()):
            log.write('{}: {}\n'.format(str(k), str(v)))
            print("{}: {}".format(str(k), str(v)))
        log.write('-' * 50 + 'End' + '-' * 50)
        print('-' * 50 + 'End' + '-' * 50)
        log.close()


def model_namer(**elements):
    name = ''
    for k, v in elements.items():
        name += str(k) + '_' + str(v)
    return name


def make_dir(dataset_name=None, model_name=None, type='checkpoints'):
    assert dataset_name in ['Cityscapes']
    if type == 'checkpoints':
        assert model_name, "model_name keyword should be specified for type='checkpoints'"
        if not os.path.isdir('./checkpoints'):
            os.mkdir('./checkpoints')
            os.mkdir(os.path.join('./checkpoints', dataset_name))
            os.mkdir(os.path.join('./checkpoints', dataset_name, 'Image'))
            os.mkdir(os.path.join('./checkpoints', dataset_name, 'Image/Training'))
            os.mkdir(os.path.join('./checkpoints', dataset_name, 'Image/Training', model_name))
            os.mkdir(os.path.join('./checkpoints', dataset_name, 'Image/Test'))
            os.mkdir(os.path.join('./checkpoints', dataset_name, 'Image/Test', model_name))
            os.mkdir(os.path.join('./checkpoints', dataset_name, 'Model'))
            os.mkdir(os.path.join('./checkpoints', dataset_name, 'Model', model_name))

        elif os.path.isdir('./checkpoints'):
            print("checkpoints directory already exists.")

    else:
        """
        for other type of directory
        """
        pass


def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid


def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d, affine=False)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=True)

    return layer


def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid. Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer









class Manager(object):
    def __init__(self, opt):
        self.opt = opt

    @staticmethod
    def report_loss(package):
        print("Epoch: {} [{:.{prec}}%]  D_loss: {:.{prec}}  G_loss: {:.{prec}}".format(package['Epoch'],
                                                package['current_step']/package['total_step']*100,
                                                package['D_loss'], package['G_loss'], prec=4))

    @staticmethod
    def adjust_dynamic_range(data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                        np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def tensor2image(self, image_tensor):
        np_image = image_tensor.squeeze().cpu().float().numpy()
        # assert np_image.shape[0] in [1, 3], print("The channel is ", np_image.shape)
        np_image = np.transpose(np_image, (1, 2, 0))  # HWC
        np_image = self.adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def save_image(self, image_tensor, path):
        np_image = self.tensor2image(image_tensor)
        pil_image = Image.fromarray(np_image)
        pil_image.save(path, self.opt.image_mode)

    def save(self, package, path, image=False, model=False):
        if image:
            self.save_image(package['target_tensor'], path)
            self.save_image(package['generated_tensor'], path)

        elif model:
            torch.save(package['G_state_dict'])
            torch.save(package['D_state_dict'])

    def __call__(self, current_step, package):
        if current_step % self.opt.display_freq == 0:
            path =
            self.save(package, path, pathimage=True)

        if current_step % self.opt.report_freq == 0:
            self.report_loss()

        if current_step % self.opt.save_freq == 0:
            path =
            self.save(package, path, model=True)


def update_lr(old_lr, n_epoch_decay, D_optim, G_optim):
    delta_lr = old_lr/n_epoch_decay
    new_lr = old_lr - delta_lr

    for param_group in D_optim.param_groups:
        param_group['lr'] = new_lr

    for param_group in G_optim.param_groups:
        param_group['lr'] = new_lr

    print("Learning rate has been updated from {} to {}.".format(old_lr, new_lr))

    return new_lr


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.detach().normal_(0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:
        module.weight.detach().normal(1.0, 0.02)
        module.bias.detach().fill_(0.0)


