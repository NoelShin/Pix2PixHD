import os
import glob
import random
import torch
from torchvision import transforms as transforms
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        dataset_dir = os.path.join('./datasets', opt.dataset_name)
        format = opt.format

        if opt.dataset_name == 'Cityscapes':
            if opt.is_train:
                self.label_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Train', 'Input', 'LabelMap', '*.' + format)))
                self.instance_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Train', 'Input', 'InstanceMap', '*.' + format)))
                self.target_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Train', 'Target', '*.' + format)))

            elif not opt.is_train:
                self.label_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Test', 'Input', 'LabelMap', '*.' + format)))
                self.instance_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Test', 'Input', 'InstanceMap', '*.' + format)))
                self.target_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Test', 'Target', '*.' + format)))

        elif opt.dataset_name == 'HMI2AIA304':
            if opt.is_train:
                self.label_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Train', 'Input', '*.' + format)))
                self.target_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Train', 'Target', '*.' + format)))

            elif not opt.is_train:
                self.label_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Test', 'Input', '*.' + format)))
                self.target_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Test', 'Target', '*.' + format)))

        else:
            raise NotImplementedError("Please check dataset_name. It should be in ['Cityscapes', 'HMI2AIA304'].")

    def get_transform(self, normalize=True):
        transform_list = []

        if self.opt.half:
            transform_list += [transforms.Resize(self.opt.image_size, interpolation=Image.NEAREST)]

        if self.opt.is_train and self.coin:
            transform_list.append(transforms.Lambda(lambda x: self.__flip(x)))

        transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        return transforms.Compose(transform_list)

    @staticmethod
    def get_edges(instance_tensor):
        edge = torch.ByteTensor(instance_tensor.shape).zero_()
        edge[:, :, 1:] = edge[:, :, 1:] | (instance_tensor[:, :, 1:] != instance_tensor[:, :, :-1])
        edge[:, :, :-1] = edge[:, :, :-1] | (instance_tensor[:, :, 1:] != instance_tensor[:, :, :-1])
        edge[:, 1:, :] = edge[:, 1:, :] | (instance_tensor[:, 1:, :] != instance_tensor[:, :-1, :])
        edge[:, :-1, :] = edge[:, :-1, :] | (instance_tensor[:, 1:, :] != instance_tensor[:, :-1, :])

        return edge.float()

    @staticmethod
    def __flip(x):
        return x.transpose(Image.FLIP_LEFT_RIGHT)

    def encode_input(self, label_tensor, instance_tensor=None):
        if self.opt.dataset_name == 'Cityscapes':
            max_label_index = 35
            shape = label_tensor.shape
            one_hot_shape = (max_label_index, shape[1], shape[2])
            label = torch.FloatTensor(torch.Size(one_hot_shape)).zero_()
            label = label.scatter_(dim=0, index=label_tensor.long(), src=torch.tensor(1.0))

            edge = self.get_edges(instance_tensor)

            input_tensor = torch.cat([label, edge], dim=0)

            return input_tensor

        elif self.opt.dataset_name == 'HMI2AIA304':
            return label_tensor

        else:
            raise NotImplementedError("Please check dataset_name. It should be in ['Cityscapes', 'HMI2AIA304'].")

    def __getitem__(self, index):
        if self.opt.dataset_name == 'Cityscapes':
            if self.opt.flip:
                self.coin = random.random() > 0.5

            label_array = Image.open(self.label_path_list[index])
            label_tensor = self.get_transform(normalize=False)(label_array) * 255.0

            instance_array = Image.open(self.instance_path_list[index])
            instance_tensor = self.get_transform(normalize=False)(instance_array)

            target_array = Image.open(self.target_path_list[index])
            target_tensor = self.get_transform(normalize=True)(target_array)

            input_tensor = self.encode_input(label_tensor, instance_tensor)

        elif self.opt.dataset_name == 'HMI2AIA304':
            self.coin= None
            label_array = Image.open(self.label_path_list[index])
            label_tensor = self.get_transform(normalize=True)(label_array)

            target_array = Image.open(self.target_path_list[index])
            target_tensor = self.get_transform(normalize=True)(target_array)

            input_tensor = self.encode_input(label_tensor)

        else:
            raise NotImplementedError("Please check dataset_name. It should be in ['Cityscapes', 'HMI2AIA304'].")

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.label_path_list)
