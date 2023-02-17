import argparse
import os

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:0')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    
#     parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
#                         help='where to decay lr, can be a list')
#     parser.add_argument('--lr_decay_rate', type=float, default=0.1,
#                         help='decay rate for learning rate')
#     parser.add_argument('--weight_decay', type=float, default=1e-4,
#                         help='weight decay')
#     parser.add_argument('--momentum', type=float, default=0.9,
#                         help='momentum')

    # dataset setting
    parser.add_argument('--datatype', type=str, default='tabular', choices=['tabular', 'image'])
    parser.add_argument('--dataset', type=str, default='adult', help='dataset')
    parser.add_argument('--std', type=float, default=1e-3, help='dataset')

    # model setting

#     parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--hidden', type=list, default = [128, 128])
    parser.add_argument('--feature_shape', type=int, default = 128)
    parser.add_argument('--feature', type=int, default = 64)
    

    opt = parser.parse_args("")

    # set the path according to the environment
    opt.model_path = './save/AW/{}_models'.format(opt.dataset)
    opt.tb_path = './save/AW/{}_tensorboard'.format(opt.dataset)

#     iterations = opt.lr_decay_epochs.split(',')
#     opt.lr_decay_epochs = list([])
#     for it in iterations:
#         opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'AW_{}_lr_{}_bsz_{}'.\
        format(opt.dataset, opt.learning_rate, opt.batch_size)

#     if opt.cosine:
#         opt.model_name = '{}_cosine'.format(opt.model_name)

#     # warm-up for large-batch training,
#     if opt.batch_size > 256:
#         opt.warm = True
#     if opt.warm:
#         opt.model_name = '{}_warm'.format(opt.model_name)
#         opt.warmup_from = 0.01
#         opt.warm_epochs = 10
#         if opt.cosine:
#             eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
#             opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
#                     1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
#         else:
#             opt.warmup_to = opt.learning_rate

#     opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
#     if not os.path.isdir(opt.tb_folder):
#         os.makedirs(opt.tb_folder)

#     opt.save_folder = os.path.join(opt.model_path, opt.model_name)
#     if not os.path.isdir(opt.save_folder):
#         os.makedirs(opt.save_folder)

#     if opt.dataset == 'cifar10':
#         opt.n_cls = 10
#     elif opt.dataset == 'cifar100':
#         opt.n_cls = 100
#     else:
#         raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt