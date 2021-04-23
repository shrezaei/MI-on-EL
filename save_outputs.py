from __future__ import print_function
import os
import numpy as np
import glob
import argparse

import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import bearpaw.models.cifar as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


torch_parallel = True
# torch_parallel = False

parser = argparse.ArgumentParser(description='Obtaining outputs from saved models.')
#model specific arguements
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: resnet)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
#general arguements
parser.add_argument('--models_path', type=str, default='models/resnet20/', help='Path to saved models')
parser.add_argument('-d', '--dataset', default='cifar10', type=str, choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'])
parser.add_argument('--output-type', type=str, default='logit',
                    help='Ensembling based on averaging confidence or logit (default: confidence)')

args = parser.parse_args()


def main():
    output_path = "outputs/" + args.models_path.split('/')[-2] + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    number_of_models = len(glob.glob(args.models_path + "*.pth.tar"))

    if args.dataset == "mnist":
        num_classes = 10
        dataloader = datasets.MNIST
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    elif args.dataset == "fmnist":
        num_classes = 10
        dataloader = datasets.FashionMNIST
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    elif args.dataset == "cifar10":
        num_classes = 10
        dataloader = datasets.CIFAR10
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif args.dataset == "cifar100":
        num_classes = 100
        dataloader = datasets.CIFAR100
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif args.dataset == "svhn":
        dataloader = datasets.SVHN
        num_classes = 10
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        print("This dataset is not supported in current version!")
        exit()

    if args.dataset == "svhn":
        trainset = dataloader(root='./data', split='train', download=True, transform=transform_train)
        testset = dataloader(root='./data', split='test', download=True, transform=transform_test)
    else:
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        testset = dataloader(root='./data', train=False, download=True, transform=transform_test)

    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=False)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False)

    train_output_all = np.zeros((len(trainset), num_classes * number_of_models))
    test_output_all = np.zeros((len(testset), num_classes * number_of_models))

    softmax_operation = torch.nn.Softmax(dim=1)

    model_counter = 0
    for model_path in glob.glob(args.models_path + "*.pth.tar"):
        print("Processing " + model_path)

        # Model
        if args.arch.startswith('resnext'):
            model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
        elif args.arch.startswith('densenet'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                dropRate=args.drop,
            )
        elif args.arch.startswith('wrn'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
        elif args.arch.endswith('resnet'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                block_name=args.block_name,
            )
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)

        state_dict_parallel = torch.load(model_path)['state_dict']
        if torch_parallel:
            model = torch.nn.DataParallel(model).cuda()
            model.load_state_dict(state_dict_parallel)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict_parallel.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        model.eval()

        temp_prediction = []
        temp_target = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            output = model(inputs)

            if args.output_type == "confidence":
                predictions = softmax_operation(output)
            elif args.output_type == "logit":
                predictions = output

            if torch_parallel:
                predictions = predictions.cpu().detach().numpy()
            else:
                predictions = predictions.detach().numpy()
            temp_prediction.extend(predictions)
            if model_counter == 0:
                temp_target.extend(targets)
        train_output_all[:, model_counter * num_classes:(model_counter + 1) * num_classes] = np.array(temp_prediction)
        # Store the ground truth when processing the first model
        if model_counter == 0:
            labels_train = np.array(temp_target)

        temp_prediction = []
        temp_target = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            output = model(inputs)

            if args.output_type == "confidence":
                predictions = softmax_operation(output)
            elif args.output_type == "logit":
                predictions = output

            if torch_parallel:
                predictions = predictions.cpu().detach().numpy()
            else:
                predictions = predictions.detach().numpy()
            temp_prediction.extend(predictions)
            if model_counter == 0:
                temp_target.extend(targets)
        test_output_all[:, model_counter * num_classes:(model_counter + 1) * num_classes] = np.array(temp_prediction)
        # Store the ground truth when processing the first model
        if model_counter == 0:
            labels_test = np.array(temp_target)

        model_counter += 1

    np.save(output_path + "/" + args.output_type + "_train.npy", train_output_all)
    np.save(output_path + "/" + args.output_type + "_test.npy", test_output_all)
    np.save(output_path + "/label_train.npy", labels_train)
    np.save(output_path + "/label_test.npy", labels_test)


if __name__ == '__main__':
    main()
