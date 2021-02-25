from tensorboardX import SummaryWriter

from configs import get_train_config
from models.main_model import MainModel
from util.utils import create_dataloader, print_state, display_pic

if __name__ == '__main__':
    config = get_train_config()
    dataloader = create_dataloader(config)
    print('The number of iterations per epoch = %d' % len(dataloader))
    epoch = 0
    total_iter = 0
    model = MainModel(config)
    model.setup()
    if config.continue_train:
        epoch, total_iter = model.load_networks(config)
    writer = SummaryWriter(config.board_path)

    for epoch in range(epoch, config.max_epoch + 1):
        epoch_iter = 0
        for i, data in enumerate(dataloader):
            model.train()
            model.set_input(data)
            model.optimize_parameters()

            total_iter += 1
            epoch_iter += 1

            if total_iter % config.print_freq:
                print_state(epoch, epoch_iter, total_iter, model.get_losses(), writer)

            if total_iter % config.display_freq:
                display_pic(model.get_current_visuals(), total_iter, config, writer)

        if config.lr_policy:
            model.update_learning_rate()

        if epoch % config.save_epoch_freq == 0:
            model.save_networks(config, epoch, total_iter)
