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
import torchvision.transforms as tranforms
from models.vgg import VGG
from models.mnist import MNIST
from models.resnet import ResNet50

from Inpainting_Method.net import PConvUNet, PConvUNet_7
from utils import get_loaders, evaluate_accuracy, \
    generate_masked_samples, generate_random_masked_samples, block_split, train_lr_logistic, compute_roc
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, precision_score, recall_score


def auc_calculate(values, labels, mask_times=None):
    # print('Build auc detector')
    X = values
    Y = labels
    X_train, Y_train, X_test, Y_test = block_split(X, Y)

    # build detector
    # print(f'the {mask_times} times')
    lr = train_lr_logistic(X_train, Y_train)

    # evaluate detector
    y_pred = lr.predict_proba(X_test)[:, 1]
    y_label_pred = lr.predict(X_test)

    # AUC
    fpr, tpr, auc_score = compute_roc(Y_test, y_pred)
    # precision = precision_score(Y_test, y_label_pred)
    # recall = recall_score(Y_test, y_label_pred)

    # acc = accuracy_score(Y_test, y_label_pred)
    # print('Detector inpainting %s_%d ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (args.attack, mask_times,
    #     auc_score, acc, precision, recall))

    return auc_score


def main(args):
    # demo the model accuracy
    saved_clf_dir = args.saved_clf_dir
    if os.path.exists(saved_clf_dir):
        pass
    else:
        raise os.error('no ckpt file exists')

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

    loss = nn.CrossEntropyLoss()

    saved_data_root = os.path.join(args.saved_data_root, '%s_data' % args.data_type, args.model_type)
    adv_data_root = os.path.join(saved_data_root, args.attack)

    Y_test = np.load(os.path.join(saved_data_root, 'Y_test.npy'))
    Y_pred_clean = np.load(os.path.join(saved_data_root, 'Y_pred_clean.npy'))
    Y_pred_adv = np.load(os.path.join(adv_data_root, 'Y_pred_%s.npy' % args.attack))
    inds_select = np.where(np.logical_and(np.equal(Y_pred_clean, Y_test), np.not_equal(Y_pred_adv, Y_test)))[0]

    X_clean = np.load(os.path.join(saved_data_root, 'X_clean.npy'))
    X_noisy = np.load(os.path.join(saved_data_root, 'X_noisy.npy'))
    X_adv = np.load(os.path.join(adv_data_root, 'X_%s.npy' % args.attack))

    # # test the acc in clean, noisy, adv
    if args.state == 'test':
        testloader_clean = DataLoader(TensorDataset(torch.tensor(X_clean), torch.tensor(X_clean), torch.tensor(Y_test)),
                                      batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_acc_clean, _ = evaluate_accuracy(testloader_clean, clf_model, loss, data_all=3, device=device)
        testloader_noisy = DataLoader(TensorDataset(torch.tensor(X_noisy), torch.tensor(X_noisy), torch.tensor(Y_test)),
                                      batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_acc_noisy, _ = evaluate_accuracy(testloader_noisy, clf_model, loss, data_all=3, device=device)
        testloader_adv = DataLoader(TensorDataset(torch.tensor(X_adv), torch.tensor(X_adv), torch.tensor(Y_test)),
                                    batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_acc_adv, _ = evaluate_accuracy(testloader_adv, clf_model, loss, data_all=3, device=device)
        print('test acc: \n---clean---%s\n---noisy---%s\n---%s---%s' % (
        test_acc_clean, test_acc_noisy, args.attack, test_acc_adv))

    # use the selected data clean right adv wrong to detect
    X_clean = np.load(os.path.join(saved_data_root, 'X_clean.npy'))[inds_select]
    X_noisy = np.load(os.path.join(saved_data_root, 'X_noisy.npy'))[inds_select]
    X_adv = np.load(os.path.join(adv_data_root, 'X_%s.npy' % args.attack))[inds_select]
    Y_test = np.load(os.path.join(saved_data_root, 'Y_test.npy'))[inds_select]
    testloader_clean = DataLoader(TensorDataset(torch.tensor(X_clean), torch.tensor(X_clean), torch.tensor(Y_test)),
                                  batch_size=args.batch_size, shuffle=False, num_workers=0)
    testloader_noisy = DataLoader(TensorDataset(torch.tensor(X_noisy), torch.tensor(X_noisy), torch.tensor(Y_test)),
                                  batch_size=args.batch_size, shuffle=False, num_workers=0)
    testloader_adv = DataLoader(TensorDataset(torch.tensor(X_adv), torch.tensor(X_adv), torch.tensor(Y_test)),
                                batch_size=args.batch_size, shuffle=False, num_workers=0)

    saved_inpainted_dir = os.path.join(args.saved_inpainting_dir, args.data_type, args.model_type,
                                       'ckpt_%d' % args.version,
                                       '%s_%s_1000000.pth' % (args.data_type, args.model_type))

    inpainted_model = PConvUNet(data_type=args.data_type).to(device)
    inpainted_model.load_state_dict(torch.load(saved_inpainted_dir, map_location=device)['model'], strict=False)
    inpainted_model.eval()
    saved_figures_dir = os.path.join(saved_data_root, 'Pictures')
    adv_figures_dir = os.path.join(saved_data_root, 'Pictures', '%s' % args.attack)
    if not os.path.exists(adv_figures_dir):
        os.makedirs(adv_figures_dir)

    loader_dict = {}
    loader_dict['clean_0'] = testloader_clean
    loader_dict['noisy_0'] = testloader_noisy
    loader_dict['adv_0'] = testloader_adv
    del testloader_clean, testloader_noisy, testloader_adv

    for i in range(args.mask_times):

        mask_time = i
        for data_type in ['clean', 'noisy', 'adv']:
            data_type_n = data_type + '_%d' % i
            data_loader = loader_dict[data_type_n]
            acc_sum, n = 0.0, 0
            for j, (image_original, x_composed, y_true) in enumerate(data_loader, 0):
                image_original = image_original.to(device)
                x_composed = x_composed.detach().to(device).requires_grad_()
                # n restart for the importance matrix
                # grad_input = 0
                # for restart_time in range(args.restart_n):
                #     random_mask = torch.bernoulli(torch.full(x_composed.size(), 1-args.restart_p)).to(device)
                #     # print(random_mask)
                #     # print(torch.sum(random_mask))
                #     re_image_com = (random_mask * x_composed).detach().requires_grad_()
                #     y_logit, va_grad_used = clf_model.get_inter_value(re_image_com)
                #     va_grad_used = torch.sum(va_grad_used, dim=1, keepdim=True)
                y_logit = clf_model(x_composed)
                y_prob = F.softmax(y_logit, dim=1)
                y_pred = y_prob.argmax(dim=1)
                # y_pred_1 = torch.topk(y_prob, k=4, dim=1)[1][:, -4]
                # acc_sum += (y_pred == y_true).float().sum().cpu().item()
                # n += y_true.shape[0]
                # l = -1*torch.sum(y_prob * torch.log(y_prob))
                l = loss(y_logit, y_pred)
                # grad_output = torch.autograd.grad(l, y_logit)[0]
                # grad_input = torch.autograd.grad(y_logit, x_inpainted, grad_outputs=grad_output)[0].detach().cpu().numpy()
                grad_input = torch.autograd.grad(l, x_composed)[0]
                # grad_input += va_grad_used*grad_input_ori

                x_composed = x_composed.detach().cpu().numpy()
                grad_input = grad_input.detach().cpu().numpy()

                mask, image_masked = generate_masked_samples(x_composed, grad_input, mask_ratio=args.mask_ratio,
                                                             replace_ratio=args.replace_ratio)
                # mask_random, image_masked_random = generate_random_masked_samples(x_composed, mask_ratio=args.mask_ratio)
                mask, image_masked = torch.from_numpy(mask).type(torch.float32).to(device), torch.from_numpy(
                    image_masked).type(torch.float32).to(device)
                with torch.no_grad():
                    image_inpainted, _ = inpainted_model(image_masked, mask)
                    image_inpainted = torch.clamp(image_inpainted, 0, 1)

                x_composed = torch.from_numpy(np.copy(x_composed)).type(torch.float32).to(device)
                image_composed = x_composed * mask + image_inpainted * (1 - mask)

                if j == 0:
                    if args.data_type == 'MNIST':
                        index = 0
                        image_masked_np = image_masked[index].detach().cpu().numpy()
                        mask_np = mask[index].detach().cpu().numpy()
                        image_inpainted_np = image_inpainted[index].detach().cpu().numpy()
                        image_composed_np = image_composed[index].detach().cpu().numpy()
                        del image_masked, mask
                        plt.figure()
                        plt.subplot(1, 4, 1)
                        plt.title('%s' % data_type)
                        plt.imshow(np.transpose(image_masked_np, (1, 2, 0))[:, :, 0], cmap='gray')
                        plt.subplot(1, 4, 2)
                        plt.title('mask')
                        plt.imshow(np.transpose(mask_np, (1, 2, 0))[:, :, 0], cmap='gray')
                        plt.subplot(1, 4, 3)
                        plt.title('%d_inpainted_image' % i)
                        plt.imshow(np.transpose(image_inpainted_np, (1, 2, 0))[:, :, 0], cmap='gray')
                        plt.subplot(1, 4, 4)
                        plt.title('image_composed')
                        plt.imshow(np.transpose(image_composed_np, (1, 2, 0))[:, :, 0], cmap='gray')
                        if data_type == 'adv':
                            plt.savefig(os.path.join(adv_figures_dir, '%s_%d_demo' % (args.attack, i)))
                        else:
                            plt.savefig(os.path.join(saved_figures_dir, '%s_%d_demo' % (data_type, i)))
                        plt.close()
                    else:
                        index = 0
                        image_masked_np = image_masked[index].detach().cpu().numpy()
                        mask_np = mask[index].detach().cpu().numpy()
                        image_inpainted_np = image_inpainted[index].detach().cpu().numpy()
                        image_composed_np = image_composed[index].detach().cpu().numpy()
                        del image_masked, mask
                        plt.figure()
                        plt.subplot(1, 4, 1)
                        plt.title('%s' % data_type)
                        plt.imshow(np.transpose(image_masked_np, (1, 2, 0)))
                        plt.subplot(1, 4, 2)
                        plt.title('mask')
                        plt.imshow(np.transpose(mask_np, (1, 2, 0)))
                        plt.subplot(1, 4, 3)
                        plt.title('%d_inpainted_image' % i)
                        plt.imshow(np.transpose(image_inpainted_np, (1, 2, 0)))
                        plt.subplot(1, 4, 4)
                        plt.title('image_composed')
                        plt.imshow(np.transpose(image_composed_np, (1, 2, 0)))
                        if data_type == 'adv':
                            plt.savefig(os.path.join(adv_figures_dir, '%s_%d_demo' % (args.attack, i)))
                        else:
                            plt.savefig(os.path.join(saved_figures_dir, '%s_%d_demo' % (data_type, i)))
                        plt.close()

                    # composed_images_used = image_composed.detach().cpu() * (1-args.w1) + args.w1 * image_original.detach().cpu()
                    # image_ori_1 = image_original.detach().cpu()
                if j == 0:
                    image_ori = [image_original.detach().cpu()]
                    composed_images_pre = [x_composed.detach().cpu()]
                    composed_images_used = [image_composed.detach().cpu()]
                    inpainted_images_used = [image_inpainted.detach().cpu()]
                    del image_inpainted, image_composed, _, image_original, x_composed
                else:
                    # composed_images_used = torch.cat((composed_images_used,image_composed.detach().cpu() * (1-args.w1) + args.w1 * image_original.detach().cpu()), dim=0)
                    # image_ori_1 = torch.cat((image_ori_1, image_original.detach().cpu()))
                    image_ori.append(image_original.detach().cpu())
                    composed_images_pre.append(x_composed.detach().cpu())
                    composed_images_used.append(image_composed.detach().cpu())
                    inpainted_images_used.append(image_inpainted.detach().cpu())
                    del image_inpainted, image_composed, _, image_original, x_composed

            if data_type.startswith('clean'):
                All_images_ori = torch.cat(image_ori)
                All_images_pre = torch.cat(composed_images_pre)
                All_images_in = torch.cat(inpainted_images_used)
                All_images_co = torch.cat(composed_images_used)
            else:
                All_images_ori = torch.cat((All_images_ori, torch.cat(image_ori)))
                All_images_pre = torch.cat((All_images_pre, torch.cat(composed_images_pre)))
                All_images_in = torch.cat((All_images_in, torch.cat(inpainted_images_used)))
                All_images_co = torch.cat((All_images_co, torch.cat(composed_images_used)))

            loader_dict['%s_%d' % (data_type, i + 1)] = DataLoader(TensorDataset(torch.cat(image_ori),
                                                                                 torch.cat(composed_images_used) * (
                                                                                             1 - args.w1) + args.w1 * torch.cat(
                                                                                     image_ori), torch.tensor(Y_test)),
                                                                   batch_size=args.batch_size, shuffle=False,
                                                                   num_workers=0)
            del loader_dict[
                '%s_%d' % (data_type, i)], image_ori, composed_images_pre, inpainted_images_used, composed_images_used

        # detect part
        #    calculate the logits part
        clf_model.eval()
        All_images_loader = DataLoader(TensorDataset(All_images_pre, All_images_in, All_images_co),
                                       batch_size=args.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for count, (image_pre, image_in, image_co) in enumerate(All_images_loader, 0):
                image_pre, image_in, image_co = image_pre.to(device), image_in.to(device), image_co.to(device)
                image_logit = clf_model(image_pre)
                inpainted_logit = clf_model(image_in)
                composed_logit = clf_model(image_co)
                if count == 0:
                    # All_image_logits = image_logit.detach().cpu().numpy()
                    All_image_logits = [image_logit.detach().cpu().numpy()]
                    All_inpainted_logits = [inpainted_logit.detach().cpu().numpy()]
                    All_composed_logits = [composed_logit.detach().cpu().numpy()]
                else:
                    # All_image_logits = np.concatenate((All_image_logits, image_logit.detach().cpu().numpy()),axis=0)
                    All_image_logits.append(image_logit.detach().cpu().numpy())
                    All_inpainted_logits.append(inpainted_logit.detach().cpu().numpy())
                    All_composed_logits.append(composed_logit.detach().cpu().numpy())

        All_image_logits = np.concatenate(All_image_logits)
        All_inpainted_logits = np.concatenate(All_inpainted_logits)
        All_composed_logits = np.concatenate(All_composed_logits)
        #   calculate the values(features) part
        print('The data length of input, inpainted and composed images', len(All_image_logits),
              len(All_inpainted_logits), 'and', len(All_composed_logits))

        l1 = np.var(All_image_logits, axis=1).reshape((-1, 1))
        l2 = np.var(All_inpainted_logits, axis=1).reshape((-1, 1))
        l3 = np.var(All_composed_logits, axis=1).reshape((-1, 1))
        if mask_time == 0:
            l2 = np.concatenate((l1, l2), axis=1)

        # calculate the pictures distance l4 and l5
        # pic_all = np.concatenate((X_clean, X_noisy, X_adv), axis=0)
        # pic_all_inp = np.concatenate((X_clean_inpainted, X_noisy_inpainted, X_adv_inpainted), axis=0)
        # pic_all_com = np.concatenate((X_clean_composed, X_noisy_composed, X_adv_composed), axis=0)
        l4 = np.sum(np.abs(np.subtract(All_images_pre.numpy(), All_images_in.numpy())), axis=(1, 2, 3)).reshape(
            (len(All_images_pre), -1))
        l5 = np.sum(np.abs(np.subtract(All_images_pre.numpy(), All_images_co.numpy())), axis=(1, 2, 3)).reshape(
            (len(All_images_pre), -1))

        # total variation loss
        if mask_time == 0:
            loss_ori = np.mean(np.abs(All_images_pre[:, :, :, 1:].numpy() - All_images_pre[:, :, :, :-1].numpy()),
                               axis=(1, 2, 3)).reshape((-1, 1)) + np.mean(
                np.abs(All_images_pre[:, :, 1:, :].numpy() - All_images_pre[:, :, :-1, :].numpy()),
                axis=(1, 2, 3)).reshape((-1, 1))
            loss_2_c = np.mean(np.abs(All_images_co[:, :, :, 1:].numpy() - All_images_co[:, :, :, :-1].numpy()),
                               axis=(1, 2, 3)).reshape((-1, 1)) + np.mean(
                np.abs(All_images_co[:, :, 1:, :].numpy() - All_images_co[:, :, :-1, :].numpy()),
                axis=(1, 2, 3)).reshape((-1, 1))
            loss_c = np.concatenate((loss_ori, loss_2_c), axis=1)
        else:
            loss_c = np.mean(np.abs(All_images_co[:, :, :, 1:].numpy() - All_images_co[:, :, :, :-1].numpy()),
                             axis=(1, 2, 3)).reshape((-1, 1)) + np.mean(
                np.abs(All_images_co[:, :, 1:, :].numpy() - All_images_co[:, :, :-1, :].numpy()),
                axis=(1, 2, 3)).reshape((-1, 1))

        # l: distance between clean and inpainted
        l1_d_all_1 = np.sum(np.abs(np.subtract(All_inpainted_logits, All_image_logits)), axis=1).reshape(
            (len(All_inpainted_logits), -1))
        l2_d_all_1 = la.norm(All_image_logits - All_inpainted_logits, axis=1).reshape((len(All_image_logits), -1))
        # 2: distance between clean and composed
        l1_d_all_2 = np.sum(np.abs(np.subtract(All_composed_logits, All_image_logits)), axis=1).reshape(
            (len(All_inpainted_logits), -1))
        l2_d_all_2 = la.norm(All_image_logits - All_composed_logits, axis=1).reshape((len(All_image_logits), -1))

        values_co = np.concatenate((l1_d_all_2, l2_d_all_2, l5), axis=1)
        values_in = np.concatenate((l1_d_all_1, l2_d_all_1, l4), axis=1)
        values_all = np.concatenate((l1_d_all_1, l2_d_all_1, l1_d_all_2, l2_d_all_2, l2, l3, l4, l5, loss_c), axis=1)
        # if args.dynamic_data == 'co':
        #     values = np.concatenate((l1_d_all_2, l2_d_all_2, l5),axis=1)
        # elif args.dynamic_data == 'in':
        #     values = np.concatenate((l1_d_all_1, l2_d_all_1, l4),axis=1)
        # else:
        #     values = np.concatenate((l1_d_all_1, l2_d_all_1, l1_d_all_2, l2_d_all_2, l2,l3,l4,l5, loss_c), axis=1)
        # values = np.concatenate((l1_d_all_1, l2_d_all_1, l1_d_all_2, l2_d_all_2,l2,l3, l4, l5, loss_c),axis=1)
        labels = np.concatenate((np.zeros(int(2 * len(values_all) / 3)), np.ones(int(len(values_all) / 3))), axis=0)

        global detection_values_all, detection_values_co, detection_values_in
        if mask_time == 0:
            detection_values_all = values_all
            detection_values_co = values_co
            detection_values_in = values_in
        else:
            detection_values_all = np.concatenate((detection_values_all, values_all), axis=1)
            detection_values_co = np.concatenate((detection_values_co, values_co), axis=1)
            detection_values_in = np.concatenate((detection_values_in, values_in), axis=1)

        values_list = ['co', 'in', 'all']
        for type in values_list:
            if type == 'co':
                values = values_co
                detection_values = detection_values_co
            elif type == 'in':
                values = values_in
                detection_values = detection_values_in
            else:
                values = values_all
                detection_values = detection_values_all

            auc_score = auc_calculate(values, labels, mask_times=mask_time + 1)
            auc_all_score = auc_calculate(detection_values, labels, mask_times=mask_time + 1)

            print('the %s  %d mask time, the auc: %.4f, the all auc: %.4f' % (
            type, mask_time + 1, auc_score, auc_all_score))
            with open('./%s/%s_data/%s/message_1.txt' % (args.saved_data_root, args.data_type, args.model_type),
                      'a') as file:
                file.write('%s features, %s %d mask time, the auc: %.4f, the all auc: %.4f' % (
                type, args.attack, mask_time + 1, auc_score, auc_all_score))
                file.write(' \n')
                file.close()

        # for combination used
        # if mask_time+1 == 10:
        #     file_dir = './Methods_Comparison/LLNA/data/%s/%s'%(args.data_type, args.model_type)
        #     if not os.path.exists(file_dir):
        #         os.makedirs(file_dir)
        #     labels = labels.reshape((-1,1))
        #     values_all = np.concatenate((detection_values_all, labels), axis=1)
        #     np.save(os.path.join(file_dir, 'LLNA_%s_%d.npy'%(args.attack, mask_time+1)), values_all)

    if mask_time == 1000:
        print('too high the mask times')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inpainting Images Generation')
    parser.add_argument('--data-type', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100', 'MNIST'])
    parser.add_argument('--model-type', default='VGG19', type=str, choices=['VGG19', 'ResNet50', 'MNIST'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--state', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--mask_ratio', default=0.03, type=float)
    parser.add_argument('--replace_ratio', default=0.03, type=float)
    parser.add_argument('--mask_times', default=10, type=int)
    parser.add_argument('--w1', type=float, default=0)

    # used to calculate the importance matrix, abandoned
    # parser.add_argument('--restart_n', type=int, default=10)
    # parser.add_argument('--restart_p', type=float, default=0.1)

    parser.add_argument('--attack', default='all', type=str,
                        choices=['all', 'FGSM', 'BIM', 'PGD', 'APGD', 'CW', 'Square', 'AutoAttack', 'DeepFool'])
    # parser.add_argument('--eps', default=0.3, type=float)
    parser.add_argument('--saved_clf_dir', default='./BaseLine_Detection/saved_model', type=str)
    parser.add_argument('--saved_inpainting_dir', default='./Inpainting_Method/saved_dir')
    parser.add_argument('--saved_data_root', default='./Data', type=str)
    parser.add_argument('--version', default=6, type=int)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int)
    # parser.add_argument('--dynamic_data', default='all', type=str, choices=['all', 'co', 'in'])
    args = parser.parse_args()

    with open('%s/%s_data/%s/message.txt' % (args.saved_data_root, args.data_type, args.model_type), 'a') as file:
        file.write('{ \n')
        file.write('generate inpainting loop, generata the inpaiting samples')
        for arg in vars(args):
            print(arg, ':', getattr(args, arg))

            file.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        # file.write(acc)
        # file.write('\n')
        file.write('} \n')
        file.close()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

    warnings.filterwarnings('ignore', category=Warning)
    # Attacks = ['FGSM', 'PGD', 'CW', 'DeepFool', 'Square', 'AutoAttack']
    Attacks = ['advgan']
    if args.attack == 'all':
        for attack in Attacks:
            args.attack = attack
            # print(args.attack)
            main(args)
    else:
        main(args)