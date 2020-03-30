import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
# print(model_names)
# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 256), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)',
#                     dest='weight_decay')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')

best_acc1 = 0


def start_training_session(parsed_arguements):
    state_object = parsed_arguements
    if state_object.seed is not None:
        random.seed(state_object.seed)
        torch.manual_seed(state_object.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    return main_worker(state_object)



def main_worker(state_object):
    global best_acc1
    # create model
    if state_object.pretrained:
        print("=> using pre-trained model '{}'".format(state_object.arch))
        model = models.__dict__[state_object.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(state_object.arch))
        model = models.__dict__[state_object.arch]()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Going to Train on ","cuda" if torch.cuda.is_available() else "cpu")

    state_object.trained_on_gpu = True if torch.cuda.is_available() else False
    state_object.used_num_gpus = torch.cuda.device_count()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        if state_object.arch.startswith('alexnet') or state_object.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)
        
    
    train_with_gpu = False
    # If there is GPU support then leach everything and 
    model.to(device)

    if torch.cuda.is_available():
        train_with_gpu = True


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), state_object.learning_rate,
                                momentum=state_object.momentum,
                                weight_decay=state_object.weight_decay)

    # # todo optionally resume from a checkpoint
    # if state_object.resume:
    #     if os.path.isfile(state_object.resume):
    #         print("=> loading checkpoint '{}'".format(state_object.resume))
    #         if state_object.gpu is None:
    #             checkpoint = torch.load(state_object.resume)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(state_object.gpu)
    #             checkpoint = torch.load(state_object.resume, map_location=loc)
    #         state_object.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if state_object.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(state_object.gpu)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(state_object.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(state_object.resume))
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(state_object.dataset_final_path, 'train')
    valdir = os.path.join(state_object.dataset_final_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler=None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=state_object.batch_size, shuffle=True,
        num_workers=state_object.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=state_object.batch_size, shuffle=False,
        num_workers=state_object.workers, pin_memory=True)

    if state_object.evaluate:
        validate(val_loader, model, criterion, state_object,device)
        return
    
    print("Training/Testing Datasets Loaded!")
    epoch_histories = {
        'train': [],
        'validation': []
    }
    for epoch in range(state_object.start_epoch, state_object.epochs):
        adjust_learning_rate(optimizer, epoch, state_object)
        if epoch % 2 == 0:
            print("Training Epoch : ",epoch)
        # train for one epoch
        train_history = train(train_loader, model, criterion, optimizer, epoch, state_object,device)
        epoch_histories['train'].append(train_history)
        # evaluate on validation set
        acc1,validation_history = validate(val_loader, model, criterion, state_object,device)
        epoch_histories['validation'].append(validation_history)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # todo : add checkpointing of best model.
        # if not state_object.multiprocessing_distributed or (state_object.multiprocessing_distributed
        #         and state_object.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': state_object.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best)
    return epoch_histories

def train(train_loader, model, criterion, optimizer, epoch, state_object,device):
    history = {
        'loss': [],
        'accuracy':[],
        'batch_time':[]
    }
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        history['accuracy'].append(float(top1.avg))
        history['loss'].append(float(losses.avg))
        history['batch_time'].append(float(batch_time.avg))
        
        if i % state_object.print_frequency == 0:
            progress.display(i)
    
    return history


def validate(val_loader, model, criterion, state_object,device):
    history = {
        'loss': [],
        'accuracy':[],
        'batch_time':[]
    }
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device,non_blocking=True)
            target = target.to(device,non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            history['accuracy'].append(float(top1.avg))
            history['loss'].append(float(losses.avg))
            history['batch_time'].append(float(batch_time.avg))

            if i % state_object.print_frequency == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg,history


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, state_object):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = state_object.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()