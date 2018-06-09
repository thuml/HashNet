def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)

    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1

    return optimizer


def step_lr_scheduler(param_lr, optimizer, iter_num, gamma, step, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (gamma ** (iter_num // step))

    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1

    return optimizer


def multistep_lr_scheduler(param_lr, optimizer, init_lr, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = init_lr
    elif epoch < 120:
        lr = init_lr * 0.1
    else:
        lr = init_lr * 0.01

    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return optimizer

schedule_dict = {"inv":inv_lr_scheduler, "step":step_lr_scheduler, "multistep":multistep_lr_scheduler}
