import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# from torch.utils.data import Sampler

import faiss
import numpy as np

from imagenet_class_labels import labels_dict

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# args of the original script
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--n-threads', default=1, type=int, metavar='N',
                    help='number of threads for add ans search (default: 1)')
parser.add_argument('--patch-size', default=8, type=int,
        help='size of the unfolded patches')
parser.add_argument('--patch-stride', default=1, type=int,
        help='stride of the unfolded patches')
parser.add_argument('--image-size', default=128, type=int,
        help='size of the image')
parser.add_argument('-k', '--k-neighbors', nargs='+',  default=[1], type=int, help='number of nearest neighbors')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--classes', nargs='+', default=[0, 1], type=int,
                    metavar='N', help='classes on which nearest neighbors is performed')
parser.add_argument('--index-type', default='flat', type=str,
                    metavar='N', help='type of index (flat or flat_voronoi)')
parser.add_argument('--n-voronoi-cells', default=1000, type=int,
                    metavar='N', help='number of vornoi cells for index')
parser.add_argument('--patch-decision-method', nargs='+', default=['vote'], type=str,
                    metavar='N', help='method to decide wether a patch belongs to a class or not')


class SubsetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def get_imagenet_class_indices(classes):
    train_indices = []
    val_indices = []

    imagenet_train_class_indices = np.load('./imagenet_train_class_indices.npy').astype('int')
    imagenet_val_class_indices = np.load('./imagenet_val_class_indices.npy').astype('int')

    for cl in classes:
        train_indices.append(np.arange(imagenet_train_class_indices[cl], imagenet_train_class_indices[cl+1]).astype('int'))
        val_indices.append(np.arange(imagenet_val_class_indices[cl], imagenet_val_class_indices[cl+1]).astype('int'))

    return np.concatenate(train_indices), np.concatenate(val_indices)


def decision(ids, n_patches_per_class, method='vote'):
    n_classes = n_patches_per_class.shape[0]
    n_patches, k_neighbors = ids.shape

    img_class = np.zeros(n_classes, dtype='int')
    if method == 'vote':
        for i_patch in range(n_patches):
            patch_class = np.zeros(n_classes, dtype='int')
            for k in range(k_neighbors):
                for i_class in range(n_classes):
                    if ids[i_patch, k] < n_patches_per_class[i_class]:
                        patch_class[i_class] += 1
                        break
            patch_class = patch_class.argmax()
            img_class[patch_class] += 1
    elif method == 'all':
        for i_patch in range(n_patches):
            patch_class = np.zeros(n_classes, dtype='int')
            for k in range(k_neighbors):
                for i_class in range(n_classes):
                    if ids[i_patch, k] < n_patches_per_class[i_class]:
                        patch_class[i_class] += 1
                        break
            unique_patch_class = np.unique(np.argwhere(patch_class > 0))
            if len(unique_patch_class) > 1:
                continue
            img_class[unique_patch_class[0]] += 1

    return img_class


def main():
    global args
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.n_threads)


    dimension = 3 * args.patch_size**2
    unfold_func = torch.nn.Unfold(kernel_size=args.patch_size, stride=args.patch_stride)
    def unfold(x):
        return unfold_func(x).transpose(1, 2) # shape batch_size, n_patches, 3*args.patch_size**2

    print('Building index of type {} ...'.format(args.index_type))

    if args.index_type == 'flat':
        index = faiss.IndexFlatL2(dimension)
    elif args.index_type == 'flat_voronoi':
        nlist = args.n_voronoi_cells
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.nprobe = 1

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ]))


    n_classes = len(args.classes)

    # get all patches to train the index
    if args.index_type in ['flat_voronoi']:
        print('Creating voronoi cells...')
        train_indices, _ = get_imagenet_class_indices(args.classes)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=SubsetSampler(train_indices))

        patches = None
        start_index = 0
        for i, (input, target) in enumerate(train_loader):
            if target[0].item() not in args.classes:
                raise Exception('train loader not working, got target of class {} not in {}'.format(target[0].item(), args.classes))

            input_patches = unfold(input).contiguous().view(-1, dimension).numpy()

            if patches is None:
                patches = np.zeros((1300*n_classes*(input_patches.shape[0]//input.size(0)), dimension), dtype='float32') #FIXME

            patches[start_index:start_index+input_patches.shape[0], :] = input_patches
            start_index += input_patches.shape[0]

        index.train(patches)
        del patches

    print('Training nearest neighbors...')
    train_start_time = time.time()

    n_patches_per_class = np.zeros(n_classes, dtype='int64')

    n_patches_total = 0
    for i_class, class_number in enumerate(args.classes):
        class_train_indices, _ = get_imagenet_class_indices([class_number])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=SubsetSampler(class_train_indices))

        n_inputs = 0
        class_start_time = time.time()
        for i, (input, target) in enumerate(train_loader):
            if target[0].item() != class_number:
                raise Exception('Loader not working, got target of class {} != {}'.format(target[0].item(), class_number))

            input_patches = unfold(input).contiguous().view(-1, dimension).numpy()

            index.add(input_patches)

            n_inputs += input.size(0)
            n_patches_total += input_patches.shape[0]


        print('  Class {}, {} samples done in {:.0f}s.'.format(class_number, n_inputs, time.time() - class_start_time))
        n_patches_per_class[i_class] = n_patches_total

    print('Training, {}M patches done in {:0f}s.'.format(n_patches_total // 1000000, time.time() - train_start_time))

    print('Validate...')
    validation_start_time = time.time()

    accuracies = np.zeros((len(args.k_neighbors), len(args.patch_decision_method)))
    class_accuracies = np.zeros((len(args.k_neighbors), len(args.patch_decision_method), n_classes))

    n_inputs_total = 0
    for i_class, class_number in enumerate(args.classes):
        _, class_val_indices = get_imagenet_class_indices([class_number])
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=SubsetSampler(class_val_indices))

        n_inputs = 0
        class_start_time = time.time()
        for i, (input, target) in enumerate(val_loader):
            if target[0].item() !=  class_number:
                raise Exception('Loader not working, got target of class {} != {}'.format(target[0].item(), class_number))

            n_inputs += input.size(0)

            input_patches = unfold(input).contiguous().view(-1, dimension).numpy()

            distances, ids = index.search(input_patches, max(args.k_neighbors))

            for i_k, k in enumerate(args.k_neighbors):
                for i_method, method in enumerate(args.patch_decision_method):
                    ids_ = ids[:,:k]

                    img_class = decision(ids_, n_patches_per_class, method=method)

                    predicted_class = args.classes[img_class.argmax()]
                    classif_right = int(target[0].item()) == predicted_class
                    accuracies[i_k, i_method] += int(classif_right)
                    class_accuracies[i_k, i_method, i_class] += int(classif_right)

        class_accuracies[:, :, i_class] /= n_inputs
        n_inputs_total += n_inputs
        print('Class {} ({}), {} samples, done in {:.0f}s.'.format(class_number, labels_dict[class_number], n_inputs, time.time() - class_start_time))

    print('Validation time : {:0f}s'.format(time.time() - validation_start_time))

    accuracies /= n_inputs_total

    np.save('accuracies_k_{}_method_{}_classes_{}.npy'.format(args.k_neighbors, args.patch_decision_method, args.classes), accuracies)
    np.save('class_accuracies_k_{}_method_{}_classes_{}.npy'.format(args.k_neighbors, args.patch_decision_method, args.classes), class_accuracies)

    for i_k, k in enumerate(args.k_neighbors):
        for i_method, method in enumerate(args.patch_decision_method):
            print('k = {}, method {}, accuracy: {:.1f}%'.format(k, method, 100*accuracies[i_k, i_method]))
            for i_class, cl in enumerate(args.classes):
                print('  - class {} ({}), accuracy: {:.1f}%'.format(cl, labels_dict[cl], 100*class_accuracies[i_k, i_method, i_class]))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


if __name__ == '__main__':
    main()
