import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F
from torchvision import datasets, transforms
import torchvision

from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from scipy.spatial.distance import cdist
from scipy import ndimage
from sklearn import svm


#注意这里的标准差很奇怪，有两个情况
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# Data
def get_loaders(data_type, batch_size, num_cpus=0):
    if data_type == 'CIFAR10':
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./BaseLine_Detection/data/downloaded', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_cpus)

        testset = torchvision.datasets.CIFAR10(
            root='./BaseLine_Detection/data/downloaded', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_cpus)
    elif data_type == 'MNIST':
        print('Preparing MNIST data')
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.MNIST(
            root='./BaseLine_Detection/data/downloaded', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_cpus)

        testset = torchvision.datasets.MNIST(
            root='./BaseLine_Detection/data/downloaded', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_cpus)
    elif data_type == 'CIFAR100':
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./BaseLine_Detection/data/downloaded', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_cpus)

        testset = torchvision.datasets.CIFAR100(
            root='./BaseLine_Detection/data/downloaded', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_cpus)
    else:
        trainloader, testloader = None, None

    return trainloader, testloader


def get_loaders_EPS(data_type, batch_size, num_cpus=0):
    if data_type == 'CIFAR10':
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./BaseLine_Detection/data/downloaded', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_cpus)

        testset = torchvision.datasets.CIFAR10(
            root='./BaseLine_Detection/data/downloaded', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_cpus)

    return trainloader, testloader

def evaluate_accuracy(data_iter, model, loss, data_all=2, device=None):
    if device is None and isinstance(model, torch.nn.Module):
        # 如果没指定device就使用model的device
        device = list(model.parameters())[0].device
    acc_sum , n = 0.0, 0
    test_l_sum, batch_count = 0.0, 0
    model.eval()
    with torch.no_grad():
        if data_all==2:
            for x, y in data_iter:
                x = x.to(device)
                y = y.to(device)
                model.eval()
                y_pred = model(x)
                acc_sum += (y_pred.argmax(dim=1) == y).float().sum().cpu().item()
                l = loss(y_pred, y)
                test_l_sum += l.cpu().item()
                n += y.shape[0]
                batch_count += 1
                del x
        elif data_all ==3:
            for x, z, y in data_iter:
                x = x.to(device)
                y = y.to(device)
                model.eval()
                y_pred = model(x)
                acc_sum += (y_pred.argmax(dim=1) == y).float().sum().cpu().item()
                l = loss(y_pred, y)
                test_l_sum += l.cpu().item()
                n += y.shape[0]
                batch_count += 1
                del x
        else:
            raise os.error('the data in dataloader is not right')

    return acc_sum/n, test_l_sum/batch_count

def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < 0.99)[0]
    # assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = 1.

    return np.reshape(x, original_shape)

def get_noisy_samples(X_test, device= torch.device('cuda:0')):
    """
    TODO : Scale of this noisy function is not very right, default = 8/255
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    # if attack in ['jsma']:
    #     X_test_noisy = np.zeros_like(X_test)
    #     for i in range(len(X_test)):
    #         # Count the number of pixels that are different
    #         nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
    #         # Randomly flip an equal number of pixels (flip means move to max
    #         # value of 1)
    #         X_test_noisy[i] = flip(X_test[i], nb_diff)
    # else:
        # Add Gaussian noise to the samples
        # X_test_noisy = np.minimum(
        #     np.maximum(
        #         X_test + np.random.normal(loc=0, scale=8/255,
        #                                   size=X_test.shape),
        #         0
        #     ),
        #     1
        # )
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_noisy = torch.minimum(
        torch.maximum(
            X_test + torch.rand(X_test.shape, device=device)* 8/255,
            torch.tensor(0).to(device)
        ),
        torch.tensor(1).to(device)
    )
    # X_test_noisy = torch.FloatTensor(X_test_noisy,)

    return X_test_noisy

# def generate_masked_samples(x_input, batch_grads, mask_ratio=0.2):
#     a_batches = np.max(batch_grads, axis=1)
#     a_batches = np.broadcast_to(a_batches[:, np.newaxis, :, :], (len(x_input), 3,32,32))
#     grads = batch_grads/a_batches
#     grads = np.sum(grads, axis=1, keepdims=True)/3
#     ratio = int(mask_ratio * 100)
#     thresholds = np.percentile(grads,ratio, axis=(2,3))
#     thresholds = np.broadcast_to(thresholds[:, :, np.newaxis, np.newaxis], (thresholds.shape[0], thresholds.shape[1], 32,32))
#     masks_all = np.where(grads>thresholds, 1,0)
#     masks_all = np.broadcast_to(masks_all, (len(masks_all), 3, 32,32))
#     data_masked_all = masks_all * x_input
            
#     return masks_all, data_masked_all

def generate_masked_samples(x_input, batch_grads, mask_ratio=0.2, replace_ratio=None):
    # a_batches = np.max(batch_grads, axis=1)
    # a_batches = np.broadcast_to(a_batches[:, np.newaxis, :, :], (len(x_input), 3,32,32))
    # grads = batch_grads/a_batches
    # grads = np.sum(grads, axis=1, keepdims=True)/3
    grads = batch_grads
    mask_ratio = mask_ratio * 100
    thresholds = np.percentile(grads, mask_ratio, axis=(1, 2, 3))
    # print(thresholds[:3])
    thresholds = np.broadcast_to(thresholds[:, np.newaxis, np.newaxis, np.newaxis],
                                 (thresholds.shape[0], x_input.shape[1], x_input.shape[2], x_input.shape[3]))
    # print(thresholds[:3])
    masks_all = np.where(grads > thresholds, 1.0, 0.0)
    # print(np.sum(masks_all))

    # masks_all += np.random.uniform(0,1,size=masks_all.shape)
    # masks_all = np.where(masks_all>0.5, 1, 0)
    # # print(np.sum(masks_all))
    # num_to_convert = int(0.5 * np.sum(masks_all == 0))
    # indices_to_convert = np.where(masks_all == 0)
    # indices_to_convert = np.random.choice(indices_to_convert
    #                                       , num_to_convert, replace=False)
    # print(np.sum(masks_all))
    # masks_all[indices_to_convert] = 1
    # print(np.sum(masks_all))

    data_masked_all = masks_all * x_input

    # replace_ratio = int(replace_ratio*100)
    # replace_thresholds = np.percentile(grads, replace_ratio, axis=(1,2,3))
    # replace_thresholds = np.broadcast_to(replace_thresholds[:, np.newaxis, np.newaxis, np.newaxis], (replace_thresholds.shape[0], x_input.shape[1], x_input.shape[2],x_input.shape[3]))
    # replace_masks_all = np.where(grads>replace_thresholds, 1, 0)
    # masks_all = np.broadcast_to(masks_all, (len(masks_all), x_input.shape[1], x_input.shape[2],x_input.shape[3]))
    # select the mini
    # grads_2 = np.abs(batch_grads)
    # ratio_2 = int(mask_ratio * 100 /2 + 1)
    # thresholds_2 = np.percentile(grads_2, ratio_2, axis=(1,2,3))
    # thresholds_2 = np.broadcast_to(thresholds_2[:, np.newaxis, np.newaxis, np.newaxis], (thresholds_2.shape[0], x_input.shape[1], x_input.shape[2],x_input.shape[3]))
    # masks_all_2 = np.where(grads_2>thresholds_2, 1, 0)
    # masks_all = masks_all * masks_all_2

    return masks_all, data_masked_all

def generate_random_masked_samples(x_input, mask_ratio=0.2):
    random_masks = np.random.choice([0,1], size=x_input.shape, p=[mask_ratio, 1-mask_ratio])
    data_masked_all = random_masks * x_input

    return random_masks, data_masked_all


def block_split(X, Y):
    """
    Split the data into 80% for training and 20% for testing
    in a block size of 100.
    :param X:
    :param Y:
    :return:
    """
    print("Isolated split 80%, 20%, for training and testing")
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_norm, Y_norm = X[:partition], Y[:partition]
    X_noisy, Y_noisy = X[partition: 2*partition], Y[partition: 2*partition]
    X_adv, Y_adv = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.008) * 100

    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def train_lr_logistic(X,Y):
    lr = LogisticRegressionCV(max_iter=1000).fit(X,Y)
    return lr

def train_lr_kernel(X, Y):
    lr = svm.SVC(kernel='rbf', probability=True).fit(X,Y)
    return lr

def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    return fpr, tpr, auc_score

def File_Record(file_path, message):
    # 记录每次跑出结果，寻找最优参数
    # str_1 = []
    # file_1 = open(file_path, mode='r')
    # for line in file_1.readlines():
    #     str_1.append(line.replace('\n', ''))
    # file_1.close()
    # if message in str_1:
    #     pass
    # else:
    #     Record_txt = open(file_path, mode='a')
    #     Record_txt.write(message)
    #     Record_txt.write('\n')
    #     Record_txt.close()
    Record_txt = open(file_path, mode='a')
    Record_txt.write(message)
    Record_txt.write('\n')
    Record_txt.close()

    return 0

## FS methods used
def reduce_precision_np(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    npp_int = npp - 1
    x_int = np.rint(x * npp_int)
    x_float = x_int / npp_int
    return x_float

def median_filter_np(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    return ndimage.filters.median_filter(x, size=(1,1,width,height), mode='reflect')


# KDBU methods used
def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]

def get_deep_representations(model, X, model_type, batch_size=128):
    """
    TODO
    :param model:
    :param X:
    :param batch_size:
    :return:
    use the object attribute
    """
    if model_type == 'VGG19':
        output_dim = model.fc2.out_features
    elif model_type == 'ResNet50':
        output_dim = model.linear.in_features
    elif model_type == 'MNIST':
        output_dim = model.fc2.in_features

    # else:
    #     output_dim = None
    #     print('Some thing wrong with get_deep_representations')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            output[i * batch_size:(i + 1) * batch_size] = \
                model.deep_features(torch.tensor(X[i * batch_size:(i + 1) * batch_size], dtype=torch.float).to(device)).cpu().detach().numpy()
    return output

def get_mc_predictions(model, X, model_type, nb_iter=50, batch_size=128):
    """
    TODO
    :param model:
    :param X:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    # if model_type == 'VGG19':
    #     output_dim = model.output_dim
    # elif model_type == 'ResNet50':
    #     output_dim = model.output_dim
    output_dim = model.output_dim
    # else:
    #     print('Some thing wrong with get_mc_predictions')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        model.eval()
        for i in range(n_batches):
            output[i * batch_size:(i + 1) * batch_size] = \
                model.mc_features(torch.tensor(X[i * batch_size:(i + 1) * batch_size], dtype=torch.float).to(device)).cpu().detach().numpy()
        return output
    preds_mc = []
    with torch.no_grad():
        for i in tqdm(range(nb_iter)):
            preds_mc.append(predict())

    return np.asarray(preds_mc)

def merge_kd(densities_pos, densities_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = densities_neg.reshape((1, -1)).transpose([1, 0])
    values_pos = densities_pos.reshape((1, -1)).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    # lr = LogisticRegressionCV().fit(values, labels)

    return values, labels

def merge_kdbu(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    # lr = LogisticRegressionCV().fit(values, labels)

    return values, labels

def load_characteristics(method ,attack, k, model_type):
    """
    Load multiple characteristics for one dataset and one attack.
    :param dataset:
    :param attack:
    :param characteristics:
    :return:
    """
    X, Y = None, None

    # print("  -- %s" % characteristics)
    file_name = '\CIFAR10\%s\data\%s\%s_%s.npy'%(method, model_type, attack, str(k))
    data = np.load(file_name)
    if data.ndim == 1:
        X = data.reshape((-1,1))
        test_number = int(len(data)/3)
        Y = np.concatenate([np.zeros(2*test_number), np.ones(test_number)]).reshape((-1,1))
    else:
        if X is None:
            X = data[:, :-1]
        else:
            X = np.concatenate((X, data[:, :-1]), axis=1)
        if Y is None:
            Y = data[:, -1]  # labels only need to load once


    return X, Y

def get_np_data(dataset, batch_size):
    """
    get the numpy data type of data in float32
    :param dataset:
    :param batch_size:
    :return:
    """
    assert dataset in ['CIFAR10', 'CIFAR100', 'MNIST'], 'dataset parameter must be either "CIFAR10" "CIFAR100" or "Tiny_ImagNet" '
    if dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./BaseLine_Detection/data/downloaded', train=True,
                                                download=True,
                                                transform=transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(cifar10_mean, cifar10_std),
            ]))
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='./BaseLine_Detection/data/downloaded', train=False,
                                               download=True,
                                               transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(cifar10_mean, cifar10_std),
            ]))
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

    elif dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='./BaseLine_Detection/data/downloaded', train=True,
                                              download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR100(root='./BaseLine_Detection/data/downloaded', train=False,
                                             download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    elif dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./BaseLine_Detection/data/downloaded',train=True, download=True,
                                             transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./BaseLine_Detection/data/downloaded', train=False,download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    else:
        print('No dataset is available in this project')

    X_train = []
    Y_train =[]
    for i, (x_train, y_train) in enumerate(trainloader, 0):
        if i == 0:
            print(x_train.dtype, x_train.shape, y_train.dtype, y_train.shape)  # torch.float32
            x_train, y_train = x_train.numpy(), y_train.numpy()  # transform to npy data type
            X_train, Y_train = x_train, y_train
        else:
            x_train, y_train = x_train.numpy(), y_train.numpy()
            X_train = np.concatenate((X_train, x_train), axis=0)
            Y_train = np.concatenate((Y_train, y_train), axis=0)
    X_test = []
    Y_test = []
    for i, (x_test, y_test) in enumerate(testloader, 0):
        if i == 0:
            x_test, y_test = x_test.numpy(), y_test.numpy()  # transform to npy data type
            X_test, Y_test = x_test, y_test
        else:
            x_test, y_test = x_test.numpy(), y_test.numpy()
            X_test = np.concatenate((X_test, x_test), axis=0)
            Y_test = np.concatenate((Y_test, y_test), axis=0)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train).astype(np.float32)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test).astype(np.float32)

    return X_train, Y_train, X_test, Y_test

def score_samples(kdes, samples, preds):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :return:
    """
    results = []
    for x, i in tqdm(zip(samples, preds)):
        kde = kdes[i]
        result = kde.score_samples(np.reshape(x,(1,-1)))
        results.append(result)
    results = np.asarray(results)

    return results

## LID
def get_lid(model, X_test, X_test_noisy, X_test_adv, model_type, k=20, batch_size=128):
    """
    Get local intrinsic dimensionality
    :param model:
    :param X_train:
    :param Y_train:
    :param X_test:
    :param X_test_noisy:
    :param X_test_adv:
    :return: artifacts: positive and negative examples with lid values,
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(model, X_test, X_test_noisy,
                                                              X_test_adv, k=k, batch_size=batch_size, model_type=model_type)
    print("lids_normal:", lids_normal.shape)
    print("lids_noisy:", lids_noisy.shape)
    print("lids_adv:", lids_adv.shape)
    lids_pos = lids_adv
    lids_neg = np.concatenate((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels

def get_lids_random_batch(model, X, X_noisy, X_adv, model_type, k=20, batch_size=128):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model:
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images
    :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    # get deep representations
    # funcs = [K.function([pre_model.layers[0].input, K.learning_phase()], [out])
    #              for out in get_layer_wise_activations(pre_model, dataset)]
    model.eval()
    # funcs = get_layer_wise_activations(model, model_type)
    # lid_dim = len(funcs) + 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lids = []
    lids_adv = []
    lids_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))

    temp_X = torch.rand(64,3,32,32).cuda()
    if model_type == 'MNIST':
        temp_X = torch.rand(64,1,28,28).cuda()
    temp_list = model.feature_list_MD(temp_X)[1]
    num_output = len(temp_list)
    X = torch.from_numpy(X)
    X_noisy = torch.from_numpy(X_noisy)
    X_adv = torch.from_numpy(X_adv)
    for i_batch in tqdm(range(n_batches)):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        output, out_features = model.feature_list_MD(X[start:end].to(device))
        X_act = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            # out_features[i] = torch.mean(out_features[i].data, 2)
            X_act.append(np.asarray(out_features[i].cpu().detach(), dtype=np.float32).reshape((out_features[i].size(0), -1)))

        output, out_features = model.feature_list_MD(X_noisy[start:end].to(device))
        X_act_noisy = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            # out_features[i] = torch.mean(out_features[i].data, 2)
            X_act_noisy.append(np.asarray(out_features[i].cpu().detach(), dtype=np.float32).reshape((out_features[i].size(0), -1)))

        output, out_features = model.feature_list_MD(X_adv[start:end].to(device))
        X_act_adv = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            # out_features[i] = torch.mean(out_features[i].data, 2)
            X_act_adv.append(np.asarray(out_features[i].cpu().detach(), dtype=np.float32).reshape((out_features[i].size(0), -1)))

        lid_batch = np.zeros(shape=(n_feed, num_output))
        lid_batch_noisy = np.zeros(shape=(n_feed, num_output))
        lid_batch_adv = np.zeros(shape=(n_feed, num_output))
        for j in range(num_output):
            lid_score = mle_batch(X_act[j], X_act[j], k = k)
            lid_score = lid_score.reshape((lid_score.shape[0]))
            lid_batch[:, j] = lid_score
            lid_noisy_score = mle_batch(X_act[j], X_act_noisy[j], k=k)
            lid_noisy_score = lid_noisy_score.reshape((lid_noisy_score.shape[0]))
            lid_batch_noisy[:, j] = lid_noisy_score
            lid_adv_score = mle_batch(X_act[j], X_act_adv[j], k = k)
            lid_adv_score = lid_adv_score.reshape((lid_adv_score.shape[0]))
            lid_batch_adv[:,j] = lid_adv_score
        lids.extend(lid_batch)
        lids_noisy.extend(lid_batch_noisy)
        lids_adv.extend(lid_batch_adv)
    lids = np.asarray(lids, dtype=np.float32)
    lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_noisy, lids_adv

def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    if k == 0:
        a = np.array([0])
        return a
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_neg, X_pos))
    y = np.concatenate((np.zeros(X_neg.shape[0]), np.ones(X_pos.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

## MD
def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    # if assume_centered=False, then the test set is supposed to have the same mean vector as the training set
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    with torch.no_grad():
        for data, target in train_loader:
            total += data.size(0)
            data = data.clone().detach().to(device)
            output, out_features_cuda = model.feature_list_MD(data)
            out_features = []
            for features in out_features_cuda:
                out_features.append(features.to(torch.device('cpu')))

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()
            # temp_start = time.time()
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0

                    for out in out_features:
                        list_features[out_count][label] = []
                        list_features[out_count][label].append(out[i].view(1,-1))
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label].append(out[i].view(1,-1))
                        out_count += 1
                num_sample_per_class[label] += 1

            # # construct the sample matrix
            # for i in range(data.size(0)):
            #     label = target[i]
            #     if num_sample_per_class[label] == 0:
            #         out_count = 0
            #         for out in out_features:
            #             list_features[out_count][label] = out[i].view(1, -1)
            #             out_count += 1
            #     else:
            #         out_count = 0
            #         for out in out_features:
            #             list_features[out_count][label] \
            #                 = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
            #             out_count += 1
            #     num_sample_per_class[label] += 1
            # temp_end = time.time()
            # print(temp_end-temp_start)
        for layer_idx in range(len(list_features)):
            for label in range(num_classes):
                list_features[layer_idx][label] = torch.cat(list_features[layer_idx][label], dim=0)


    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        # temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        temp_list = torch.Tensor(num_classes, int(num_feature))
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    sample_class_mean_cuda = []
    for sample in sample_class_mean:
        sample = sample.cuda()
        sample_class_mean_cuda.append(sample)
    precision_cuda = []
    for precis in precision:
        precision_cuda.append(precis.cuda())

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean_cuda, precision_cuda

def get_Mahalanobis_score_adv(model, test_data, test_label, num_classes, sample_mean, precision,
                              layer_index, magnitude, model_type):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    batch_size = 64
    total = 0
    b = int(len(test_data) / batch_size)
    test_label = torch.tensor(test_label)
    test_data = torch.tensor(test_data)

    for data_index in range(int(len(test_data) / batch_size)+1):
        if total + batch_size > len(test_data):
            target = test_label[total:].cuda()
            data = test_data[total:].cuda()
        else:
            target = test_label[total: total + batch_size].cuda()
            data = test_data[total: total + batch_size].cuda()
        total += batch_size
        data, target = torch.tensor(data, requires_grad=True), torch.tensor(target)
        out_features = model.intermediate_forward_MD(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if model_type == 'MNIST':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
        # tempInputs = torch.add(data.data, -magnitude, gradient)
        tempInputs = torch.add(input=data.data, alpha=-magnitude, other=gradient)

        noise_out_features = model.intermediate_forward_MD(torch.tensor(tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

    return Mahalanobis