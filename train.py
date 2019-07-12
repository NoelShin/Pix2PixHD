if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    from torch.utils.data import DataLoader
    from networks import Discriminator, Generator, Loss
    from option import TrainOption, TestOption
    from pipeline import CustomDataset
    from utils import Manager, update_lr, weights_init
    import datetime

    torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    lr = opt.lr

    dataset = CustomDataset(opt)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.n_workers,
                             shuffle=opt.shuffle)

    test_opt = TestOption().parse()
    test_dataset = CustomDataset(test_opt)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=test_opt.batch_size,
                                  num_workers=test_opt.n_workers)

    G = Generator(opt).apply(weights_init).to(DEVICE)
    D = Discriminator(opt).apply(weights_init).to(DEVICE)

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

            input, target = input.to(DEVICE), target.to(DEVICE)

            D_loss, G_loss, target_tensor, generated_tensor = criterion(D, G, input, target)

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'D_loss': D_loss.detach().item(),
                       'G_loss': G_loss.detach().item(),
                       'D_state_dict': D.state_dict(),
                       'G_state_dict': G.state_dict(),
                       'target_tensor': target_tensor,
                       'generated_tensor': generated_tensor.detach()}

            manager(package)

            if opt.debug:
                break

        if epoch % opt.epoch_save == 0:
            torch.save(G.state_dict(), os.path.join(test_opt.model_dir, "{:d}_G.pt".format(epoch)))
            torch.save(D.state_dict(), test_opt.model_dir, "{:d}_D.pt".format(epoch))

            image_dir = os.path.join(test_opt.image_dir, "{:d}".format(epoch))
            os.makedirs(image_dir, exist_ok=True)
            with torch.no_grad():
                for i, (input, target) in enumerate(test_data_loader):
                    input = input.to(DEVICE)
                    fake = G(input)
                    manager.save_image(fake, os.path.join(image_dir, "{:d}_fake.png".format(i)))
                    manager.save_image(target, os.path.join(image_dir, "{:d}_real.png".format(i)))

        if epoch > opt.epoch_decay:
            lr = update_lr(opt.lr, lr, opt.n_epochs - opt.epoch_decay, D_optim, G_optim)

    print("Total time taken: ", datetime.datetime.now() - start_time)
