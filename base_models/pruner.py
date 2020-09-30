import logging
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
from PIL import Image
from torchvision.transforms import transforms

Image.warnings.simplefilter('ignore')
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = False
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
ngpus = 2
epochs = 100
post_epochs = 0
output_path = '/braintree/home/fgeiger/weight_initialization/base_models/model_weights/'  # os.path.join(os.path.dirname(__file__), 'model_weights/')
data_path = '/braintree/data2/active/common/imagenet_raw/' if 'IMAGENET' not in os.environ else os.environ['IMAGENET']
batch_size = 256
weight_decay = 1e-4
momentum = .9
step_size = 20
lr = .1
workers = 20
optimizer = 'SGD'
if ngpus > 0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prev = 0.0
epsilon = 0.2
running_diff = []


def train(identifier, model):
    from Utils import load, generator, metrics
    from prune import prune_loop
    logger.info('We start training the model')

    ## Random Seed and Device ##
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = load.device(1)

    ## Data ##
    dataset = 'imagenet'
    print('Loading {} dataset.'.format(dataset))
    input_shape, num_classes = load.dimension(dataset)
    prune_loader = get_dataloader()
    # load.dataloader(dataset, batch_size, True, workers, 10 * num_classes)
    # train_loader = load.dataloader(dataset, batch_size, True, workers)
    # test_loader = load.dataloader(dataset, batch_size, False, workers)

    ## Model ##
    print('Creating {} model.'.format(model))
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer('sgd')
    optimizer = opt_class(generator.parameters(model), lr=lr, weight_decay=weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=.1)

    ## Save Original ##
    torch.save(model.state_dict(), "{}/model.pt".format(output_path))
    torch.save(optimizer.state_dict(), "{}/optimizer.pt".format(output_path))
    torch.save(scheduler.state_dict(), "{}/scheduler.pt".format(output_path))

    ## Train-Prune Loop ##
    for sparsity in [0.5, 1, 1.5, 2, 2.5, 3]:
        if os.path.exists(output_path + f'{identifier}_prune{sparsity}_epoch_{epochs:02d}.pth.tar'):
            logger.info('Model already trained')
            return
        ## Prune ##
        print('Pruning with {} for {} epochs.'.format('synflow', epochs))
        params = generator.masked_parameters(model, False, True, False)
        pruner = load.pruner('synflow')(params)
        sparsity = (10 ** (-float(sparsity)))
        prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                   'exponential', 'global', epochs, False, False)
        torch.save(model.state_dict(), "{}/pruned_{}.pt".format(output_path, str(sparsity)))

        ckpt_data = {}
        ckpt_data['epoch'] = epochs
        ckpt_data['state_dict'] = model.state_dict()
        ckpt_data['optimizer'] = optimizer.state_dict()
        ckpt_data['scheduler'] = scheduler.state_dict()
        torch.save(ckpt_data, output_path +
                   f'{identifier}_prune{sparsity}_epoch_{epochs}.pth.tar')
        prune_result = metrics.summary(model,
                                       pruner.scores,
                                       metrics.flop(model, input_shape, device),
                                       lambda p: generator.prunable(p, True, False))
        total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
        possible_params = prune_result['size'].sum()
        total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
        possible_flops = prune_result['flops'].sum()
        # print("Train results:\n", train_result)
        print("Prune results:\n", prune_result)
        print(
            "Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
        print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

        # Reset Model's Weights
        original_dict = torch.load("{}/model.pt".format(output_path), map_location=device)
        original_weights = dict(filter(lambda v: (v[1].requires_grad == True), original_dict.items()))
        model_dict = model.state_dict()
        model_dict.update(original_weights)
        model.load_state_dict(model_dict)

        # Reset Optimizer and Scheduler
        optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(output_path), map_location=device))
        scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(output_path), map_location=device))


def get_dataloader():
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'train'),
        torchvision.transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             **kwargs)

    return dataloader
