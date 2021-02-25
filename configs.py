import argparse


def get_train_config():
    parser = argparse.ArgumentParser(description='3dcnn for line art colorization')
    # dataset config
    parser.add_argument('--img_path', default='/data2/wn/Video_dataset/train/frame')
    parser.add_argument('--xdog_path', default='/data2/wn/Video_dataset/train/xdog')
    parser.add_argument('--img_size', default=512)
    parser.add_argument('--seq_len', default=8)
    parser.add_argument('--threshold', default=10)
    # train config
    parser.add_argument('--name', default='3dcnn')
    parser.add_argument('--isTrain', default=True)
    parser.add_argument('--continue_train', default='')
    parser.add_argument('--checkpoints_dir', default='./checkpoints')
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--workers', default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--lr_policy', default='')
    parser.add_argument('--max_epoch', default=5000)
    parser.add_argument('--seed', default=222)
    # loss fuction config
    parser.add_argument('--beta1', default=0.5)
    parser.add_argument('--gan_mode', type=str, default='lsgan')
    parser.add_argument('--lambda_L1', default=1)
    parser.add_argument('--lambda_style', default=1000)
    parser.add_argument('--lambda_content', default=1)
    # visualization config
    parser.add_argument('--print_freq', default=1)
    parser.add_argument('--display_freq', default=100)
    parser.add_argument('--save_epoch_freq', default=5)
    parser.add_argument('--board_path', default='./board')

    return parser.parse_args()