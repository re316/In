import os
import sys
import time
import argparse
import warnings
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms
from models.vgg import VGG
from models.mnist import MNIST
from models.resnet import ResNet50

from torchattacks import FGSM, PGD, CW, APGD, Square, AutoAttack, DeepFool
from utils import get_loaders, evaluate_accuracy, block_split, train_lr_logistic, compute_roc, get_noisy_samples, File_Record

from Methods_Comparison.FS.api import detect_by_FS
from Methods_Comparison.KDBU.api import detect_by_KDBU
from Methods_Comparison.LID.api import detect_by_LID
from Methods_Comparison.MD.api import detect_by_MD
from Methods_Comparison.JTLA.api import detect_by_JTLA
from Methods_Comparison.ODD.api import detect_by_ODD
from Methods_Comparison.EPS.api import detect_by_EPS
from Methods_Comparison.LIBRE.api import detect_by_LIBRE

def main(args):
    _, test_loader = get_loaders(args.data_type, args.batch_size)
    saved_clf_dir = args.saved_clf_dir
    if os.path.exists(saved_clf_dir):
        pass
    else:
        raise os.error('no ckpt file exists')
    

    loss = nn.CrossEntropyLoss()
    
    a = []
    b = {}
    if args.attack == 'all':
        Attacks = ['FGSM', 'PGD','DeepFool','AutoAttack','Square']
    else:
        Attacks = [args.attack]
    # if args.data_type == 'CIFAR10':
    #     Attacks = ['CW']
    # else:
    #     Attacks = ['FGSM', 'PGD','CW', 'DeepFool','AutoAttack', 'Square']

    # Methods = ['KDBU', 'FS', 'LID', 'MD', "JTLA"]
    for attack in Attacks:
        if args.data_type == 'CIFAR10':
            if args.model_type == 'VGG19':
                clf_model = VGG('VGG19').to(device)
                clf_model.load_state_dict(
                    torch.load(os.path.join(saved_clf_dir, 'CIFAR10_VGG19_100.pth'), map_location=device))
            elif args.model_type == 'ResNet50':
                clf_model = ResNet50().to(device)
                clf_model.load_state_dict(
                    torch.load(os.path.join(saved_clf_dir, 'CIFAR10_ResNet50_100.pth'), map_location=device)
                )
        elif args.data_type == 'CIFAR100':
            if args.model_type == 'VGG19':
                clf_model = VGG('VGG19', output_dim=100).to(device)
                clf_model.load_state_dict(
                    torch.load(os.path.join(saved_clf_dir, 'CIFAR100_VGG19_100.pth'), map_location=device)
                )
            elif args.model_type == 'ResNet50':
                clf_model = ResNet50(output_dim=100).to(device)
                clf_model.load_state_dict(
                    torch.load(os.path.join(saved_clf_dir, 'CIFAR100_ResNet50_100.pth'), map_location=device)
                )
        elif args.data_type == 'MNIST':
            clf_model = MNIST().to(device)
            clf_model.load_state_dict(torch.load(os.path.join(saved_clf_dir, 'MNIST_100.pth'), map_location=device))
        else:
            print('not program yet')

        aucs = {}
        spend_times = {}
        aucs['attack'] = attack
        args.attack = attack
        model_type = args.model_type
        saved_data_root = os.path.join(args.saved_data_root, '%s_data'%args.data_type, args.model_type)
        adv_data_root = os.path.join(saved_data_root, args.attack)

        Y_test = np.load(os.path.join(saved_data_root,'Y_test.npy'))
        Y_pred_clean = np.load(os.path.join(saved_data_root, 'Y_pred_clean.npy'))
        Y_pred_adv = np.load(os.path.join(adv_data_root, 'Y_pred_%s.npy'%args.attack))
        # if args.attack == 'CW':
        #     Y_pred_adv = np.load(os.path.join(adv_data_root, 'Y_pred_%s_%d.npy'%(args.attack, args.cw_step)))
        inds_select = np.where(np.logical_and(np.equal(Y_pred_clean, Y_test), np.not_equal(Y_pred_adv, Y_test)))[0]

        X_clean = np.load(os.path.join(saved_data_root,'X_clean.npy'))
        X_noisy = np.load(os.path.join(saved_data_root, 'X_noisy.npy'))
        X_adv = np.load(os.path.join(adv_data_root, 'X_%s.npy'%args.attack))
        # if args.state == 'test':
        #     testloader_clean = DataLoader(TensorDataset(torch.tensor(X_clean), torch.tensor(X_clean), torch.tensor(Y_test)),
        #                         batch_size=args.batch_size,shuffle=False,num_workers=0)
        #     test_acc_clean, _ = evaluate_accuracy(testloader_clean,clf_model, loss, data_all=3, device=device)
        #     testloader_noisy = DataLoader(TensorDataset(torch.tensor(X_noisy), torch.tensor(X_noisy), torch.tensor(Y_test)),
        #                         batch_size=args.batch_size,shuffle=False,num_workers=0)
        #     test_acc_noisy, _ = evaluate_accuracy(testloader_noisy,clf_model, loss, data_all=3, device=device)
        #     testloader_adv = DataLoader(TensorDataset(torch.tensor(X_adv), torch.tensor(X_adv), torch.tensor(Y_test)),
        #                         batch_size=args.batch_size,shuffle=False,num_workers=0)
        #     test_acc_adv, _ = evaluate_accuracy(testloader_adv,clf_model, loss, data_all=3, device=device)
        #     print('test acc: \n---clean---%s\n---noisy---%s\n---%s---%s' %(test_acc_clean, test_acc_noisy,args.attack, test_acc_adv))

        X_clean = np.load(os.path.join(saved_data_root,'X_clean.npy'))[inds_select]
        X_noisy = np.load(os.path.join(saved_data_root, 'X_noisy.npy'))[inds_select]
        X_adv = np.load(os.path.join(adv_data_root, 'X_%s.npy'%args.attack))[inds_select]
        # if args.attack == 'CW':
        #     X_adv = np.load(os.path.join(adv_data_root, 'X_%s_%d.npy'%(args.attack, args.cw_step)))[inds_select]
        Y_test = np.load(os.path.join(saved_data_root,'Y_test.npy'))[inds_select]
        x_to_detect = torch.from_numpy(np.concatenate([X_clean, X_noisy, X_adv], axis=0))
        X_test = X_clean
        # FS
        depth = 6
        smooth = 2
        try:
            auc_FS, time_train_all, time_test_all, = detect_by_FS(clf_model, x_to_detect, Y_test, depth=depth, smooth=smooth, attack_method=attack, model_type=model_type, saved_data_root=saved_data_root, data_type=args.data_type)
            aucs['auc_FS'] = auc_FS
        except Exception as e:
            print(f'error {e} in FS')
        ## KDBU
        band = 1
        print('band', band)
        try:
            auc_KDBU, time_train_all, time_test_all = detect_by_KDBU(clf_model, x_to_detect, Y_test, band_width=band, attack_method=attack, model_type=model_type, saved_data_root=saved_data_root, data_type=args.data_type)
            aucs['auc_KDBU'] = auc_KDBU
        except Exception as e:
            print(f'error {e} in KDBU')
        #
        # LID
        k = 30
        print('the %d nearest we evaluate '%k)
        try:
            auc_LID, time_train_all, time_test_all = detect_by_LID(clf_model, x_to_detect, Y_test, k, train_attack=attack, test_attack=attack, model_type=model_type, saved_data_root=saved_data_root, data_type=args.data_type)
            aucs['LID'] = auc_LID
        except Exception as e:
            print(f'error {e} in LID')
        # #
        ## MD
        m = 0.0005
        print('the %.4f magnitude we use in MD detecting method'% m)
        try:
            auc_MD, time_train_all, time_test_all = detect_by_MD(clf_model, x_to_detect, Y_test, m, attack, model_type, saved_data_root, data_type=args.data_type)
            aucs['MD'] = auc_MD
        except Exception as e:
            print(f'error {e} in MD')

        ## JTLA
        try:
            auc_JTLA, time_train_all, time_test_all = detect_by_JTLA(clf_model, x_to_detect, Y_test, attack, model_type, saved_data_root, data_type=args.data_type)
            aucs['JTLA'] = auc_JTLA
        except Exception as e:
            print(f'error {e} in JTLA')

            ## EPS
        if args.data_type == 'CIFAR10':
            try:
                auc_EPS, time_train_all, time_test_all = detect_by_EPS(clf_model, x_to_detect, Y_test, attack, model_type, saved_data_root, data_type=args.data_type, inds_select=inds_select)
                aucs['EPS'] = auc_EPS
            except Exception as e:
                print(f'error {e} in EPS')

        # LIBRE
        try:
            auc_LIBRE, time_train_all, time_test_all = detect_by_LIBRE(clf_model, x_to_detect, Y_test, attack, model_type, saved_data_root, data_type=args.data_type)
            aucs['LIBRE'] = auc_LIBRE
        except Exception as e:
            print(f'error {e} in LIBRE')

        a.append(aucs)
        # a.append(spend_times)
        b['aucs'] = aucs
        # b['times'] = spend_times
        File_Record('./Data/%s_data/%s/auc_best.txt'%(args.data_type,args.model_type), str(b))
        
    print('hello world')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Other Methods')
    parser.add_argument('--data-type', default='CIFAR10', type=str, choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--model-type', default='VGG19', type=str, choices=['VGG19', 'ResNet50', 'MNIST'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--state', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--attack',default='advgan', type=str, choices=['FGSM', 'PGD', 'CW', 'APGD', 'Square', 'AutoAttack', 'DeepFool', 'advgan'])
    parser.add_argument('--eps', default=0.0314, type=float)
    # parser.add_argument('--lr-min', default=0., type=float)
    # parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--saved_clf_dir', default='./BaseLine_Detection/saved_model', type=str)
    parser.add_argument('--saved_data_root', default='./Data', type=str)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cw_step', default=500, type=int)

    parser.add_argument('--method', default='all', type=str, choices=['KDBU', 'FS', 'LID', 'MD', "JTLA"])
    args = parser.parse_args()

    with open('%s/%s_data/%s/message.txt'%(args.saved_data_root, args.data_type, args.model_type), 'a') as file:
                file.write('{ \n')
                file.write('methods comparison')
                for arg in vars(args):
                    print(arg, ':', getattr(args, arg))
                
                    file.write(arg +': '+ str(getattr(args,arg)) + '\n')
                # file.write(acc)
                # file.write('\n')
                file.write('} \n')
                file.close()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.method == 'all':
    #     for method in Methods:
    #         args.method = method
    #         if args.attack == 'all':
    #             for attack in Attacks:
    #                 args.attack = attack
    #                 main(args)
    #         else:
    #             main(args)
    # else:
    #     if args.attack == 'all':
    #         for attack in Attacks:
    #             args.attack = attack
    #             main(args)
    #     else:
    #         main(args)
    main(args)


