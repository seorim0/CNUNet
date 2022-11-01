"""
Docstring for Options
"""


class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--nepoch', type=int, default=60, help='training epochs')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')

        # train settings
        parser.add_argument('--arch', type=str, default='CNUNet', help='archtechture')
        parser.add_argument('--loss_type', type=str, default='mag+real+imag', help='loss function type')
        parser.add_argument('--loss_oper', type=str, default='l1', help='loss function operation type')
        parser.add_argument('--c1', type=int, default=1, help='coupling constant 1')
        parser.add_argument('--c2', type=int, default=1, help='coupling constant 2')
        parser.add_argument('--device', type=str, default='cuda', help='gpu or cpu')

        # network settings
        parser.add_argument('--in_ch', type=int, default=2, help='channel size for input dim')
        parser.add_argument('--mid_ch', type=int, default=32, help='channel size for middle dim')
        parser.add_argument('--out_ch', type=int, default=64, help='channel size for output dim')

        # pretrained
        parser.add_argument('--env', type=str, default='base', help='log name')
        parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained_weights')
        parser.add_argument('--pretrain_model_path', type=str, default='./log/',
                            help='path of pretrained_weights')

        # dataset
        parser.add_argument('--database', type=str, default='WSJ0', help='database')
        parser.add_argument('--fft_len', type=int, default=512, help='fft length')
        parser.add_argument('--win_len', type=int, default=400, help='window length')
        parser.add_argument('--hop_len', type=int, default=100, help='hop length')
        parser.add_argument('--fs', type=int, default=16000, help='sampling frequency')
        parser.add_argument('--chunk_size', type=int, default=48000, help='chunk size')

        parser.add_argument('--noisy_dirs_for_train', type=str,
                            default='../Dataset/WSJ0/train/noisy/',
                            help='noisy dataset addr for train')
        parser.add_argument('--noisy_dirs_for_valid', type=str,
                            default='../Dataset/WSJ0/valid/noisy/',
                            help='noisy dataset addr for valid')
        parser.add_argument('--noisy_dirs_for_test', type=str,
                            default='../Dataset/WSJ0/test/noisy/',
                            help='noisy dataset addr for valid')

        return parser
