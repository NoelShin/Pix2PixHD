import os
import torch
import glob
from pipeline import CustomDataset
from network import Generator
from option import TestOption
from utils.Manger import save_image

import datetime

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    opt = TestOption().parse()
    USE_CUDA = True if torch.cuda.is_available() else False

    dataset = CustomDataset(opt)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                              shuffle=opt.is_train, num_workers=opt.n_workers)

    G = Generator(opt)

    list_G = sorted(glob.glob(os.path.join(opt.model_dir, '*_G.pt')))

    for path_G in list_G:
        G.load_state_dict(torch.load(path_G))
        G = G.cuda() if USE_CUDA else G

        for i, input in enumerate(data_loader):
            input = input.cuda() if USE_CUDA else input
            fake = G(input)
            save_image(fake.detach(), os.path.join(opt.image_dir, 'Result_{}'.format(i)))
            print('{}/{} [{:.{prec}}%] has done.'.format(i + 1, len(data_loader), (i + 1)/len(data_loader)*100,
                                                         prec=4))

    print("Total time taken: ", datetime.datetime.now() - start_time)
