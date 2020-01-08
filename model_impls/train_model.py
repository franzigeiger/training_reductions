import glob
import io
import logging
import os
import pickle
import pprint
import shlex
import subprocess
import time

import cornet
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
import tqdm
from PIL import Image
from torch.nn import Module

Image.warnings.simplefilter('ignore')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = False
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
ngpus = 2
epochs = 20
output_path = os.path.join(os.path.dirname(__file__), 'model_weights/')
data_path = '/braintree/data2/active/common/imagenet_raw/'
batch_size = 256
weight_decay = 1e-4
momentum = .9
step_size = 32
lr =.1
workers = 32
if ngpus > 0:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_gpus(n=2):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
        stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])




def get_model(pretrained=False):
    map_location = None if ngpus > 0 else 'cpu'
    model = getattr(cornet, f'cornet_S')
    # if model.lower() == 'r':
    #     model = model(pretrained=pretrained, map_location=map_location, times=times)
    # else:
    model = model(pretrained=pretrained, map_location=map_location)

    if ngpus == 0:
        model = model.module  # remove DataParallel
    if ngpus > 0:
        model = model.cuda()
    return model


# def train_model(model):
#     # train(model, save_model_epochs=50, save_model_secs=60*100)
#     trainer = ImageNetTrain(model)
#     validator = ImageNetVal(model)
#     results = {}
#     for i in range(5):
#         results[validator.name] = validator()
#         trainer.model.train()

def train(identifier,
          model,
          restore_path=None,  # useful when you want to restart training
          save_train_epochs=.1,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=5,  # how often save model weigths
          save_model_secs=60 * 10  # how often save model (in sec)
          ):
    if os.path.exists(output_path + f'{identifier}_epoch_20.pth.tar'):
        class Wrapper(Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.module = model

        model = Wrapper(model)
        logger.info('Resore weights from stored results')
        checkpoint = torch.load(output_path + f'{identifier}_epoch_20.pth.tar',
                                map_location=lambda storage, loc: storage)  # map onto cpu
        model.load_state_dict(checkpoint['state_dict'])
        model = model.module
        return model
    logger.info('We start training the model')
    if ngpus > 1 and torch.cuda.device_count() > 1:
        set_gpus(n=2)
        logger.info('We have multiple GPUs detected')
        model = nn.DataParallel(model)
        model = model.cuda()
        # model = model.to(device)
    elif ngpus > 0 and torch.cuda.device_count() is 1:
        set_gpus(n=1)
        logger.info('We run on GPU')
        model = model.to(device)
    else:
        logger.info('No GPU detected!')

    trainer = ImageNetTrain(model)
    validator = ImageNetVal(model)

    start_epoch = 0
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(0, epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if save_val_steps is not None:

                if global_step in save_val_steps:
                    print('we do this 1')
                    results[validator.name] = validator()
                    trainer.model.train()

            if output_path is not None:
                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(output_path + f'results_{identifier}.pkl', 'wb+'))

                ckpt_data = {}
                # ckpt_data['flags'] = __dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, output_path +
                                   f'{identifier}_latest_checkpoint.pth.tar')
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, output_path +
                                   f'{identifier}_epoch_{epoch:02d}.pth.tar')

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()
    return model


def test(layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    Suitable for small image sets. If you have thousands of images or it is
    taking too long to extract features, consider using
    `torchvision.datasets.ImageFolder`, using `ImageNetVal` as an example.

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    model = get_model(pretrained=True)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        _model_feats.append(np.reshape(output, (len(output), -1)).numpy())

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    model_feats = []
    with torch.no_grad():
        model_feats = []
        fnames = sorted(glob.glob(os.path.join(data_path, '*.*')))
        if len(fnames) == 0:
            raise Exception(f'No files found in {data_path}')
        for fname in tqdm.tqdm(fnames):
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise Exception(f'Unable to load {fname}')
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            model(im)
            model_feats.append(_model_feats[time_step])
        model_feats = np.concatenate(model_feats)

    if output_path is not None:
        fname = f'CORnet-{model}_{layer}_{sublayer}_feats.npy'
        np.save(os.path.join(output_path, fname), model_feats)


class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size)
        self.loss = nn.CrossEntropyLoss()
        if ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        with torch.autograd.detect_anomaly():
            if ngpus > 0:
                inp = inp.to(device)
                # target =target.to(device)
                # inp.to('cuda')
                target = target.cuda(non_blocking=True)
            # print(inp)
            output = self.model(inp)
            # print(output)
            record = {}
            loss = self.loss(output, target)
            record['loss'] = loss.item()
            record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
            record['top1'] /= len(output)
            record['top5'] /= len(output)
            record['learning_rate'] = self.lr.get_lr()[0]
            # print(loss)
            # print(f'shape {loss.shape}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        record['dur'] = time.time() - start
        return record


class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        if ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_path, 'val_in_folders'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                if ngpus > 0:
                    inp =inp.to(device)
                    target = target.to(device)
                        # = target.cuda(non_blocking=True)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res
