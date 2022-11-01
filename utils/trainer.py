import torch
from .progress import Bar
from .scores import cal_pesq, cal_stoi


######################################################################################################################
#                                          train with single loss function                                           #
######################################################################################################################
def time_loss_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)
        loss = loss_calculator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('time', train_loss, EPOCH)

    return train_loss


def mag_loss_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)

        clean_mags, _ = model.stft(targets)
        out_mags, _ = model.stft(outputs)

        loss = loss_calculator(out_mags, clean_mags)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('mag', train_loss, EPOCH)

    return train_loss


def real_imag_loss_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    train_real_loss = 0
    train_imag_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)

        clean_specs = model.cstft(targets)
        clean_real = clean_specs[:, :opt.fft_len//2+1, :]
        clean_imag = clean_specs[:, opt.fft_len//2+1:, :]

        out_specs = model.cstft(outputs)
        out_real = out_specs[:, :opt.fft_len//2+1, :]
        out_imag = out_specs[:, opt.fft_len//2+1:, :]

        real_loss = loss_calculator(out_real, clean_real)
        imag_loss = loss_calculator(out_imag, clean_imag)

        loss = (real_loss + imag_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_real_loss += real_loss.item()
        train_imag_loss += imag_loss.item()

    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('real+imag', train_loss, EPOCH)
    writer.log_train_loss('real', train_real_loss/batch_num, EPOCH)
    writer.log_train_loss('imag', train_imag_loss/batch_num, EPOCH)

    return train_loss


######################################################################################################################
#                                          train with multiple loss function                                         #
######################################################################################################################
def time_mag_loss_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    train_time_loss = 0
    train_mag_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)

        clean_mags, _ = model.stft(targets)
        out_mags, _ = model.stft(outputs)

        mag_loss = loss_calculator(out_mags, clean_mags)
        time_loss = loss_calculator(outputs, targets)

        r1 = opt.c1
        r2 = opt.c2
        r = r1 + r2
        loss = (r1 * mag_loss + r2 * time_loss) / r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_time_loss += time_loss.item()
        train_mag_loss += mag_loss.item()
    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('time+mag', train_loss, EPOCH)
    writer.log_train_loss('time', train_time_loss/batch_num, EPOCH)
    writer.log_train_loss('mag', train_mag_loss/batch_num, EPOCH)

    return train_loss


def mag_real_imag_loss_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    train_mag_loss = 0
    train_real_imag_loss = 0
    train_real_loss = 0
    train_imag_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)

        clean_specs = model.cstft(targets)
        clean_real = clean_specs[:, :opt.fft_len//2+1, :]
        clean_imag = clean_specs[:, opt.fft_len//2+1:, :]
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

        out_specs = model.cstft(outputs)
        out_real = out_specs[:, :opt.fft_len//2+1, :]
        out_imag = out_specs[:, opt.fft_len//2+1:, :]
        out_mags = torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-7)

        mag_loss = loss_calculator(out_mags, clean_mag)

        real_loss = loss_calculator(out_real, clean_real)
        imag_loss = loss_calculator(out_imag, clean_imag)
        real_imag_loss = (real_loss + imag_loss) / 2

        r1 = opt.c1
        r2 = opt.c2
        r = r1 + r2
        loss = (r1 * mag_loss + r2 * real_imag_loss) / r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mag_loss += mag_loss.item()
        train_real_imag_loss += real_imag_loss.item()
        train_real_loss += real_loss.item()
        train_imag_loss += imag_loss.item()
    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('mag+real+imag', train_loss, EPOCH)
    writer.log_train_loss('mag', train_mag_loss / batch_num, EPOCH)
    writer.log_train_loss('real+imag', train_real_imag_loss / batch_num, EPOCH)
    writer.log_train_loss('real', train_real_loss / batch_num, EPOCH)
    writer.log_train_loss('imag', train_imag_loss / batch_num, EPOCH)

    return train_loss


######################################################################################################################
#                                          valid with single loss function                                           #
######################################################################################################################
def time_loss_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)
            loss = loss_calculator(outputs, targets)

            valid_loss += loss

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = cal_pesq(enhanced_wavs, clean_wavs)
            stoi = cal_stoi(enhanced_wavs, clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('time', valid_loss, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi


def mag_loss_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)

            clean_mags, _ = model.stft(targets)
            out_mags, _ = model.stft(outputs)

            loss = loss_calculator(out_mags, clean_mags)

            valid_loss += loss

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = cal_pesq(enhanced_wavs, clean_wavs)
            stoi = cal_stoi(enhanced_wavs, clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('mag', valid_loss, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi


def real_imag_loss_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    valid_real_loss = 0
    valid_imag_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)

            clean_specs = model.cstft(targets)
            clean_real = clean_specs[:, :opt.fft_len//2+1, :]
            clean_imag = clean_specs[:, opt.fft_len//2+1:, :]

            out_specs = model.cstft(outputs)
            out_real = out_specs[:, :opt.fft_len//2+1, :]
            out_imag = out_specs[:, opt.fft_len//2+1:, :]

            real_loss = loss_calculator(out_real, clean_real)
            imag_loss = loss_calculator(out_imag, clean_imag)

            loss = (real_loss + imag_loss) / 2

            valid_loss += loss
            valid_real_loss += real_loss
            valid_imag_loss += imag_loss

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = cal_pesq(enhanced_wavs, clean_wavs)
            stoi = cal_stoi(enhanced_wavs, clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('real+imag', valid_loss, EPOCH)
    writer.log_valid_loss('real', valid_real_loss/batch_num, EPOCH)
    writer.log_valid_loss('imag', valid_imag_loss/batch_num, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi


######################################################################################################################
#                                          valid with multiple loss function                                         #
######################################################################################################################
def time_mag_loss_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    valid_time_loss = 0
    valid_mag_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)

            clean_mags, _ = model.stft(targets)
            out_mags, _ = model.stft(outputs)

            mag_loss = loss_calculator(out_mags, clean_mags)
            time_loss = loss_calculator(outputs, targets)

            r1 = opt.c1
            r2 = opt.c2
            r = r1 + r2
            loss = (r1 * mag_loss + r2 * time_loss) / r

            valid_loss += loss
            valid_time_loss += time_loss
            valid_mag_loss += mag_loss

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = cal_pesq(enhanced_wavs, clean_wavs)
            stoi = cal_stoi(enhanced_wavs, clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('time+mag', valid_loss, EPOCH)
    writer.log_valid_loss('time', valid_time_loss/batch_num, EPOCH)
    writer.log_valid_loss('mag', valid_mag_loss/batch_num, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi


def mag_real_imag_loss_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    valid_mag_loss = 0
    valid_real_imag_loss = 0
    valid_real_loss = 0
    valid_imag_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)

            clean_specs = model.cstft(targets)
            clean_real = clean_specs[:, :opt.fft_len//2+1, :]
            clean_imag = clean_specs[:, opt.fft_len//2+1:, :]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

            out_specs = model.cstft(outputs)
            out_real = out_specs[:, :opt.fft_len//2+1, :]
            out_imag = out_specs[:, opt.fft_len//2+1:, :]
            out_mags = torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-7)

            mag_loss = loss_calculator(out_mags, clean_mag)

            real_loss = loss_calculator(out_real, clean_real)
            imag_loss = loss_calculator(out_imag, clean_imag)
            real_imag_loss = (real_loss + imag_loss) / 2

            r1 = opt.c1
            r2 = opt.c2
            r = r1 + r2
            loss = (r1 * mag_loss + r2 * real_imag_loss) / r

            valid_loss += loss
            valid_mag_loss += mag_loss
            valid_real_imag_loss += real_imag_loss
            valid_real_loss += real_loss
            valid_imag_loss += imag_loss

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = cal_pesq(enhanced_wavs, clean_wavs)
            stoi = cal_stoi(enhanced_wavs, clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('mag+real+imag', valid_loss, EPOCH)
    writer.log_valid_loss('mag', valid_mag_loss/batch_num, EPOCH)
    writer.log_valid_loss('real+imag', valid_real_imag_loss/batch_num, EPOCH)
    writer.log_valid_loss('real', valid_real_loss/batch_num, EPOCH)
    writer.log_valid_loss('imag', valid_imag_loss/batch_num, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi
