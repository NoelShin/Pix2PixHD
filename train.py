if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    from option import TrainOption
    from pipeline import CustomDataset
    from networks import Generator, Discriminator, Loss
    from utils import weights_init, update_lr, Manager
    import datetime

    opt = TrainOption().parse()
    lr = opt.lr
    USE_CUDA = True if opt.gpu_ids != -1 else False
    dataset = CustomDataset(opt)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=opt.batch_size,
                                              num_workers=opt.n_workers,
                                              shuffle=opt.shuffle)

    G = Generator(opt).apply(weights_init)
    D = Discriminator(opt).apply(weights_init)

    if USE_CUDA:
        G = G.cuda()
        D = D.cuda()

    criterion = Loss(opt)

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    manager = Manager(opt)

    current_step = 0
    total_step = opt.n_epochs * len(data_loader)
    start_time = datetime.datetime.now()
    for epoch in range(1, opt.n_epochs + 1):
        for _, (input, target) in enumerate(data_loader):
            current_step += 1

            if USE_CUDA:
                input = input.cuda()
                target = target.cuda()

            D_loss, G_loss, target_tensor, generated_tensor = criterion(D, G, input, target)

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            print("let's count")
            b = datetime.datetime.now()
            a = G.state_dict()
            print("state dict time taken: ", datetime.datetime.now() - b)

            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'D_loss': D_loss.detach().item(),
                       'G_loss': G_loss.detach().item(),
                       'D_state_dict': D.state_dict(),
                       'G_state_dict': G.state_dict(),
                       'target_tensor': target_tensor,
                       'generated_tensor': generated_tensor}

            manager(package)

        if epoch > opt.epoch_decay:
            lr = update_lr(lr, opt.n_epochs - opt.epoch_decay, D_optim, G_optim)

    print("Total time taken: ", datetime.datetime.now() - start_time)
