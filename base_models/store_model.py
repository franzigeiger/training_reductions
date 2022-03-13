import io
import logging
import os
import shlex
import subprocess

import cornet
import pandas
import torch
import torch.utils.model_zoo
import torchvision
from PIL import Image

Image.warnings.simplefilter('ignore')
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = False
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
ngpus = 2
epochs = 1
output_path = '/braintree/home/fgeiger/weight_initialization/base_models/model_weights/'  # os.path.join(os.path.dirname(__file__), 'model_weights/')
data_path = '/braintree/data2/active/common/imagenet_raw/' if 'IMAGENET' not in os.environ else \
os.environ['IMAGENET']
batch_size = 256
weight_decay = 1e-4
momentum = .9
step_size = 20
lr = .1
workers = 20
if ngpus > 0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_gpus(n=2):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
        stdout=subprocess.PIPE, shell=True).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    print(gpus)

    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    print(f'GPUs {gpus}')
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ[
        'CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


def get_model(pretrained=False):
    map_location = None if ngpus > 0 else 'cpu'
    model = getattr(cornet, f'cornet_S')
    model = model(pretrained=pretrained, map_location=map_location)

    if ngpus == 0:
        model = model.module  # remove DataParallel
    if ngpus > 0:
        model = model.cuda()
    return model


def train(identifier,
          model,
          restore_path=None,  # useful when you want to restart training
          save_train_epochs=1,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=1,  # how often save model weigths
          save_model_secs=60 * 10,  # how often save model (in sec)
          areas=None
          ):
    ckpt_data = {}
    ckpt_data['epoch'] = 0
    ckpt_data['state_dict'] = model.state_dict()
    torch.save(ckpt_data, output_path + f'{identifier}_epoch_{00}.pth.tar')
