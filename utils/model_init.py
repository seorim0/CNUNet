# get architecture
def get_arch(opt):
    arch = opt.arch

    print('You choose ' + arch + '...')
    if arch == 'CNUNet':
        from models import CNUNet
        model = CNUNet(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                       WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)
    else:
        raise Exception("Arch error!")

    return model


# get trainer and validator (train method)
def get_train_mode(opt):
    from .trainer import time_loss_train, mag_loss_train, real_imag_loss_train, \
        time_mag_loss_train, mag_real_imag_loss_train
    from .trainer import time_loss_valid, mag_loss_valid, real_imag_loss_valid, \
        time_mag_loss_valid, mag_real_imag_loss_valid
    loss_type = opt.loss_type

    print('You choose ' + loss_type + '...')
    if loss_type == 'time':  # single loss function
        trainer = time_loss_train
        validator = time_loss_valid
    elif loss_type == 'mag':  # single loss function
        trainer = mag_loss_train
        validator = mag_loss_valid
    elif loss_type == 'real+imag':  # single loss function
        trainer = real_imag_loss_train
        validator = real_imag_loss_valid
    elif loss_type == 'time+mag':  # multiple(joint) loss function
        trainer = time_mag_loss_train
        validator = time_mag_loss_valid
    elif loss_type == 'mag+real+imag':  # multiple(joint) loss function
        trainer = mag_real_imag_loss_train
        validator = mag_real_imag_loss_valid
    else:
        raise Exception("Loss type error!")

    return trainer, validator


def get_loss(opt):
    from torch.nn import L1Loss
    from torch.nn.functional import mse_loss
    loss_oper = opt.loss_oper

    print('You choose loss operation with ' + loss_oper + '...')
    if loss_oper == 'l1':
        loss_calculator = L1Loss()
    elif loss_oper == 'l2':
        loss_calculator = mse_loss
    else:
        raise Exception("Arch error!")

    return loss_calculator
