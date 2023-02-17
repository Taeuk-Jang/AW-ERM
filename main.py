
import pandas as pd
import numpy as np
import time
import string, shutil

import scipy as scp
import sklearn
import sys, os, pickle

import numpy as np
import matplotlib.pyplot as plt

from src.dataloader import *
from src.model_util import *
from src.metric_util import *
from src.parser import parse_option as parser
from src.logger import Logger

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.utils as vutils
import torchvision.models as models


from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
import seaborn as sns
from sklearn.metrics import roc_auc_score

import argparse

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score


from verb_classification.data_loader import ImSituVerbGender
from verb_classification.model import *
import logging


def main():
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--resume', type = bool, default = True)
    
    parser.add_argument('--crt_type', type = str, default = 'multiclass')
    parser.add_argument('--train_type', type = str, default = 'weight')
    '''
    train type
    weight : AW-ERM
    ARL
    baseline
    '''
    parser.add_argument('--task_type', type = str, default = 'action')
    parser.add_argument('--dataset', type = str, default = 'imsitu')
    
#     args.device = 'cuda:1'


    #hyperparameters

    parser.add_argument('--start_epoch', type = int, default = 1)
    parser.add_argument('--num_epochs', type = int, default = 50)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--lr_w', type = float, default = 1e-4)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--soft', type = float, default = 0.1)
    
    parser.add_argument('--finetune', type = bool, default = True)
    
    parser.add_argument('--log_prefix', type = str, default = '')
    
    args = parser.parse_args()
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    
    if not args.crt_type == 'baseline':
        args.save_dir = args.dataset+ '/'  + args.log_prefix + '_' + args.task_type + '_' + str(args.soft) + '_' + str(args.num_epochs) + '_' + str(args.lr) + '_' + str(args.lr_w) + '_' + args.train_type + '_' \
        + args.crt_type + '/' + timestr
        args.log_dir = args.dataset+ '/'  + args.log_prefix + '_' + args.task_type + '_' + str(args.soft) + '_' + str(args.num_epochs) + '_' + str(args.lr) + '_' + str(args.lr_w) + '_' + args.train_type + '_' \
        + args.crt_type + '/' + timestr
    else:
        args.save_dir = args.dataset+ '/'  + args.log_prefix + '_' + str(args.num_epochs) + '_' + str(args.lr) + '_' + args.crt_type + '/' + timestr
        args.log_dir = args.dataset+ '/'  + args.log_prefix + '_' + str(args.num_epochs) + '_' + str(args.lr) + '_' + args.crt_type + '/' + timestr
    
    if args.dataset == 'imsitu':
        args.annotation_dir = './verb_classification/data'
        args.image_dir = './verb_classification/data/of500_images_resized'
    
        args.image_size = 256
        args.crop_size = 224
    
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

        val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])
        
    elif args.dataset == 'celeba':
        args.annotation_dir = './data/celeba'

        args.image_size = 128
        args.crop_size = 178

        normalize = transforms.Normalize(mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5])

        train_transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        normalize])

        val_transform = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        normalize])
        
        
    if args.task_type == 'action':
        args.hidden_layers = [1024, 1024]
    else:
        args.hidden_layers = [1024, 512]
        
    args.hidden_layers_weight = []

    if args.dataset == 'imsitu':
        train_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'train', transform = train_transform)

        val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'val', transform = val_transform)
    
        args.id2verb = train_data.id2verb
        
    elif args.dataset == 'celeba':   
        train_data = CelebALoader(args, annotation_dir = args.annotation_dir, \
            split = 'train', transform = train_transform)

        val_data = CelebALoader(args, annotation_dir = args.annotation_dir, \
            split = 'val', transform = val_transform)        


    args.train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
            shuffle = True, num_workers = 6, pin_memory = True)
    
    args.val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
            shuffle = False, num_workers = 4, pin_memory = True)


    args.save_dir = os.path.join('./models', args.save_dir)
    
    if os.path.exists(args.save_dir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.save_dir))
        return
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
        
        
    args.log_dir = os.path.join('./logs', args.log_dir)
    train_log_dir = os.path.join(args.log_dir, 'train')
    val_log_dir = os.path.join(args.log_dir, 'val')
    if not os.path.exists(train_log_dir): os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir): os.makedirs(val_log_dir)
    args.train_logger = Logger(train_log_dir)
    args.val_logger = Logger(val_log_dir)
    
#     logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]  %(message)s', filename = args.log_dir + '/log_' + timestr + '.txt')

    setup_logger('log_train', train_log_dir + '/log_' + \
                 '_' + str(args.num_epochs) + '_' + str(args.lr) + '_' + str(args.lr_w) + '_' + args.train_type + '_' + \
                 args.crt_type + '_' + timestr + '.txt')
    setup_logger('log_valid', val_log_dir + '/log_' + \
                 '_' + str(args.num_epochs) + '_' + str(args.lr) + '_' + str(args.lr_w) + '_' + args.train_type + '_' + \
                 args.crt_type + '_' + timestr + '.txt')
    setup_logger('log_train_verbose', train_log_dir + '/log_' + \
                 '_' + str(args.num_epochs) + '_' + str(args.lr) + '_' + str(args.lr_w) + '_' + args.train_type + '_' + \
                 args.crt_type + '_' + timestr + '_verbose.txt', verbose = False)
    setup_logger('log_valid_verbose', val_log_dir + '/log_' + \
                 '_' + str(args.num_epochs) + '_' + str(args.lr) + '_' + str(args.lr_w) + '_' + args.train_type + '_' + \
                 args.crt_type + '_' + timestr + '_verbose.txt', verbose = False)
    
    args.logger_train = logging.getLogger('log_train')
    args.logger_valid = logging.getLogger('log_valid')
    args.logger_train_verbose = logging.getLogger('log_train_verbose')
    args.logger_train_verbose.setLevel(logging.INFO)
    args.logger_valid_verbose = logging.getLogger('log_valid_verbose')
    args.logger_valid_verbose.setLevel(logging.INFO)

#     stderrLogger=logging.StreamHandler()
#     stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
#     logging.getLogger().addHandler(stderrLogger)
    


    args.encoder = Encoder('resnet50')
    args.model = VerbClassification(args, 'classifier')
    args.classifier = Classifier(args.encoder, args.model)
    
    
    print('######################')
    print(args.task_type)
    
    if not args.train_type == 'baseline':
        print('start training reweiging model')
        
        args.w_model = VerbClassification(args, 'ARL') if args.train_type == 'ARL' else VerbClassification(args, 'weighing')
        
    else:
        print('start training baseline model')
    
    args.optim_act = torch.optim.Adam(args.classifier.parameters(), args.lr, weight_decay = 1e-4)
    
    if not args.train_type == 'baseline':
        args.optim_w = torch.optim.Adam(args.w_model.parameters(), args.lr_w, weight_decay = 1e-3)
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        args.classifier = nn.DataParallel(args.classifier)
        if not args.train_type == 'baseline':
            args.w_model = nn.DataParallel(args.w_model)
    
    args.classifier = args.classifier.cuda()
    if not args.train_type == 'baseline':
        args.w_model = args.w_model.cuda()
    
#     args.criterion = nn.CrossEntropyLoss().cuda()
    if args.crt_type == 'multilabel':
        args.criterion = ML_criterion
    elif args.crt_type == 'multiclass':
        args.criterion = MC_criterion
    elif args.crt_type == 'ce':
        args.criterion = nn.CrossEntropyLoss().cuda()
        
    args.ref_criterion = nn.CrossEntropyLoss(reduce = False).cuda()
        
    best_performance = 0
    args.train_log_dict = {'pred_score':[],'target':[],'gender':[],'pred_exact':[],'weight':[],'dialog':[], 'loss':[],\
                          'acc top5':[], 'acc top5_male':[], 'acc top5_female':[], 'acc top10':[], 'acc top10_male':[], \
                           'fscore':[], 'fscore_male':[], 'fscore_female':[],\
                           'acc top10_female':[], 'acc':[], 'acc_male':[], 'acc_female':[], 'mAP':[], 'mAP_male':[], 'mAP_female':[]}
    args.valid_log_dict = {'pred_score':[],'target':[],'gender':[],'pred_exact':[],'weight':[],'dialog':[], 'loss':[],\
                          'acc top5':[], 'acc top5_male':[], 'acc top5_female':[], 'acc top10':[], 'acc top10_male':[], \
                           'fscore':[], 'fscore_male':[], 'fscore_female':[],\
                           'acc top10_female':[], 'acc':[], 'acc_male':[], 'acc_female':[], 'mAP':[], 'mAP_male':[], 'mAP_female':[]}
    
    
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, epoch)
        
        current_performance = test(args, epoch)
        
        is_best = current_performance > best_performance
        best_performance = max(current_performance, best_performance)
        
        model_state = {
            'epoch': epoch + 1,
            'state_dict_enc': args.encoder.state_dict(),
            'state_dict_act': args.model.state_dict(),
            'best_performance': best_performance}
        if not args.train_type == 'baseline':
            model_state['state_dict_w'] = args.w_model.state_dict()
            
        save_checkpoint(args, model_state, is_best, os.path.join(args.save_dir, \
                'checkpoint.pth.tar'))

        # at the end of every run, save the model
        if epoch == args.num_epochs:
            torch.save(model_state, os.path.join(args.save_dir, \
                'checkpoint_%s.pth.tar' % str(args.num_epochs)))
            
        with open(os.path.join(args.log_dir, 'train', 'eval.pickle'.format(epoch)), 'wb') as handle:
            pickle.dump(args.train_log_dict, handle)
        with open(os.path.join(args.log_dir, 'val', 'eval.pickle'.format(epoch)), 'wb') as handle:
            pickle.dump(args.valid_log_dict, handle)
    
def train(args, epoch):
    
    args.classifier.train()
    
    if not args.train_type == 'baseline':
        args.w_model.train()
        
    loss_logger_before_w = AverageMeter()
    loss_logger_after_w =  AverageMeter()
    
    t = tqdm(args.train_loader, desc = 'Train %d' % epoch)
    
    nProcessed = 0
    
    res = list()
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        if args.dataset == 'celeba' and batch_idx > 500:
            t.close()
            break
        
        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()
        
        ####### train classifier #######
        preds, features = args.classifier(images)
        
        if not args.train_type == 'baseline':
            if args.task_type == 'action':
#                 weight = args.w_model(features, targets.float()).detach()
                weight = args.w_model(features, targets).detach()
            elif args.task_type == 'gender':
                weight = args.w_model(features, genders).detach()

            weight = 1 + weight / torch.mean(weight, 0, keepdim = True)
            
        elif args.train_type == 'baseline' and not args.crt_type == 'ce':
            weight = torch.ones_like(preds).cuda()

        # compute loss and add softmax to preds (crossentropy loss integrates softmax)
        if args.crt_type == 'multilabel':
            loss = args.criterion(preds, targets, weight.detach())
        elif args.crt_type == 'multiclass':
            if args.task_type == 'action':
                loss, _ = args.criterion(preds, targets, weight.detach(), args.soft)
            elif args.task_type == 'gender':
                loss, _ = args.criterion(preds, genders, weight.detach(), args.soft)
        elif args.crt_type == 'ce':
            loss = args.criterion(preds, targets.max(1, keepdim=False)[1])
            
        loss_logger_before_w.update(loss.item())
        
        # backpropogation
        args.optim_act.zero_grad()
        loss.backward()
        args.optim_act.step()
        
        if not args.train_type == 'baseline':
            ####### train adversary #######
            preds, features = args.classifier(images)
            
            if args.task_type == 'action':
#                 weight = args.w_model(features.detach(), targets.float())
                weight = args.w_model(features.detach(), targets.float())
            elif args.task_type == 'gender':
                weight = args.w_model(features.detach(), genders.float())
            
            weight = 1 + weight / torch.mean(weight, 0, keepdim = True)

            # compute loss and add softmax to preds (crossentropy loss integrates softmax)
            if args.crt_type == 'multilabel':
                loss = args.criterion(preds.detach(), targets, weight)
            elif args.crt_type == 'multiclass':
                if args.task_type == 'action':
                    loss, _ = args.criterion(preds.detach(), targets, weight, args.soft)
                elif args.task_type == 'gender':
                    loss, _ = args.criterion(preds.detach(), genders, weight, args.soft)
                
            elif args.crt_type == 'ce':
                loss = args.criterion(preds, targets.max(1, keepdim=False)[1])

            loss_logger_after_w.update(loss.item())

            # backpropogation
            args.optim_w.zero_grad()
            loss.backward()
            args.optim_w.step()

        if args.task_type == 'action':
            with torch.no_grad():
                sample_loss = args.ref_criterion(preds, targets.argmax(-1))
        else:
            with torch.no_grad():
                sample_loss = args.ref_criterion(preds, genders.argmax(-1))
        
        preds = F.softmax(preds, dim=1)
        preds_max = preds.max(1, keepdim=True)[1]

        # save the exact preds (binary)
        tensor = torch.tensor((), dtype=torch.float64)
        preds_exact = tensor.new_zeros(preds.size())
        
        for idx, item in enumerate(preds_max):
            preds_exact[idx, item] = 1

        res.append((image_ids, preds.detach().cpu(), targets.detach().cpu(), genders.detach().cpu(), \
                    preds_exact, weight.detach().cpu(), sample_loss.detach().cpu()))

        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger_after_w.avg, completed = nProcessed)
    
    total_preds   = torch.cat([entry[1] for entry in res], 0).numpy()
    total_targets = torch.cat([entry[2] for entry in res], 0).numpy() if args.task_type == 'action' else torch.cat([entry[3] for entry in res], 0).numpy()
    total_genders = torch.cat([entry[3] for entry in res], 0).numpy()
    total_preds_exact = torch.cat([entry[4] for entry in res], 0).numpy()
    total_weight = torch.cat([entry[5] for entry in res], 0).numpy()
    total_loss = torch.cat([entry[6] for entry in res], 0).numpy()

    acc = np.mean(total_targets.argmax(-1) == total_preds_exact.argmax(-1))
    acc_male = np.mean(total_targets.argmax(-1)[total_genders[:, 0].nonzero()] == total_preds_exact.argmax(-1)[total_genders[:, 0].nonzero()])
    acc_female = np.mean(total_targets.argmax(-1)[total_genders[:, 1].nonzero()] == total_preds_exact.argmax(-1)[total_genders[:, 1].nonzero()])
    acc_diff = abs(acc_male - acc_female)
    
    
    if args.task_type == 'action' and args.dataset == 'imsitu':
        cnt = 0
        top5 = total_preds.argsort(-1)[:, -5:]
        match_t5 = np.zeros_like(total_targets.argmax(-1))

        for i, label in enumerate(total_targets.argmax(-1).reshape(-1,1)):
            if label in top5[i]:
                match_t5[i] = 1
        #         cnt += 1

        acc_t5 = np.mean(match_t5)
        acc_t5_male = np.mean(match_t5[total_genders[:, 0].nonzero()])
        acc_t5_female = np.mean(match_t5[total_genders[:, 1].nonzero()])
        acc_t5_diff = abs(acc_t5_female - acc_t5_male)

        cnt = 0
        top10 = total_preds.argsort(-1)[:, -10:]
        match_t10 = np.zeros_like(total_targets.argmax(-1))

        for i, label in enumerate(total_targets.argmax(-1).reshape(-1,1)):
            if label in top10[i]:
                match_t10[i] = 1
        #         cnt += 1

        # acc_t10 = cnt/(i + 1)
        acc_t10 = np.mean(match_t10)
        acc_t10_male = np.mean(match_t10[total_genders[:, 0].nonzero()])
        acc_t10_female = np.mean(match_t10[total_genders[:, 1].nonzero()])
        acc_t10_diff = abs(acc_t10_female - acc_t10_male)

#     print(total_targets.shape)
#     print(total_preds_exact.shape)
    task_f1_score = f1_score(total_targets, total_preds_exact, average = 'macro')
    task_f1_score_male = f1_score(total_targets[total_genders[:, 0].nonzero()], total_preds_exact[total_genders[:, 0].nonzero()], average = 'macro')
    task_f1_score_female = f1_score(total_targets[total_genders[:, 1].nonzero()], total_preds_exact[total_genders[:, 1].nonzero()], average = 'macro')
    task_f1_score_diff = abs(task_f1_score_male - task_f1_score_female)

    man_idx = total_genders[:, 0].nonzero()[0]
    woman_idx = total_genders[:, 1].nonzero()[0]

    preds_man = total_preds[man_idx]
    preds_woman = total_preds[woman_idx]
    targets_man = total_targets[man_idx]
    targets_woman = total_targets[woman_idx]
    
    meanAP = average_precision_score(total_targets, total_preds, average='macro')
    meanAP_man = average_precision_score(targets_man, preds_man, average='macro')
    meanAP_woman = average_precision_score(targets_woman, preds_woman, average='macro')

#     print('loss', loss_logger.avg, epoch)
    args.logger_train.info('######## Train epoch  : {} ########, '.format(epoch))
    
    args.logger_train.info('acc : {:.3f}'.format(acc))
    args.logger_train.info('acc diff: {:.3f}'.format(acc_diff))
    
    if args.task_type == 'action' and args.dataset == 'imsitu':
        args.logger_train.info('acc top5 : {:.3f}'.format(acc_t5))
        args.logger_train.info('acc top5 diff : {:.3f}'.format(acc_t5_diff))
        args.logger_train.info('acc top10 : {:.3f}'.format(acc_t10))
        args.logger_train.info('acc top10 diff : {:.3f}'.format(acc_t10_diff))

    # print('loss : {:.3f}'.format(loss_logger.avg, epoch))
    args.logger_train.info('task_f1_score : {:.3f}'.format(task_f1_score, ))
    args.logger_train.info('task_f1_score diff: {:.3f}'.format(task_f1_score_diff, ))


    args.logger_train.info('loss : {:.3f}'.format(loss_logger_before_w.avg, epoch))
    args.logger_train.info('meanAP : {:.3f}'.format(meanAP, epoch))
    args.logger_train.info('meanAP_man : {:.3f}'.format(meanAP_man, epoch))
    args.logger_train.info('meanAP_woman : {:.3f}'.format(meanAP_woman, epoch))

    args.logger_train.info('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    args.logger_train.info('Train epoch  : {}, meanAP: {:.2f}, meanAP_man: {:.2f}, meanAP_woman: {:.2f}'.format( \
        epoch, meanAP*100, meanAP_man*100, meanAP_woman*100))
    
    if args.task_type == 'action' and args.dataset == 'imsitu':
        dialog, diag_dict = gen_eval_diag(args.id2verb, total_preds_exact, total_targets, man_idx, woman_idx)
        args.logger_train_verbose.info(dialog)

        args.train_log_dict['dialog'].append(diag_dict)
        
        args.train_log_dict['acc top5'].append(acc_t5)
        args.train_log_dict['acc top5_male'].append(acc_t5_male)
        args.train_log_dict['acc top5_female'].append(acc_t5_female)

        args.train_log_dict['acc top10'].append(acc_t10)
        args.train_log_dict['acc top10_male'].append(acc_t10_male)
        args.train_log_dict['acc top10_female'].append(acc_t10_female)
        
    args.train_log_dict['pred_score'].append(total_preds)
    args.train_log_dict['target'].append(total_targets)
    args.train_log_dict['gender'].append(total_genders)
    args.train_log_dict['pred_exact'].append(total_preds_exact)
    args.train_log_dict['weight'].append(total_weight)
    args.train_log_dict['loss'].append(total_loss)
    
    args.train_log_dict['acc'].append(acc)
    args.train_log_dict['acc_male'].append(acc_male)
    args.train_log_dict['acc_female'].append(acc_female)
    
    args.train_log_dict['fscore'].append(task_f1_score)
    args.train_log_dict['fscore_male'].append(task_f1_score_male)
    args.train_log_dict['fscore_female'].append(task_f1_score_female)
    
    args.train_log_dict['mAP'].append(meanAP)
    args.train_log_dict['mAP_male'].append(meanAP_man)
    args.train_log_dict['mAP_female'].append(meanAP_woman)

#         with open(os.path.join(args.log_dir, 'train', 'eval_{}.pickle'.format(epoch)), 'wb') as handle:
#             pickle.dump(diag_dict, handle)
    
def test(args, epoch):

    # set the eval mode
    args.classifier.eval()
    
    if not args.train_type == 'baseline':
        args.w_model.eval()
        
    nProcessed = 0
    loss_logger = AverageMeter()
    
#     nVal = len(args.val_loader.dataset) # number of images

    res = list()
    t = tqdm(args.val_loader, desc = 'Val %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()
        
        # Forward, Backward and Optimize
        preds, features = args.classifier(images)
        
        
        if not args.train_type == 'baseline':
            
            if args.task_type == 'action':
#                 weight = args.w_model(features.detach(), targets.float())
                weight = args.w_model(features.detach(), targets.float())
            elif args.task_type == 'gender':
                weight = args.w_model(features.detach(), genders.float())
                    
            #case 1
            weight = 1 + weight / torch.mean(weight, 0, keepdim = True)
            #case 2
#             weight = 1 + weight / torch.mean(weight, 1, keepdim = True)

        elif args.train_type == 'baseline' and not args.crt_type == 'ce':
            weight = torch.ones_like(preds).cuda()
            
        if args.crt_type == 'multilabel':
            loss = args.criterion(preds.detach(), targets, weight)
        elif args.crt_type == 'multiclass':
            if args.task_type == 'action':
                loss, _ = args.criterion(preds.detach(), targets, weight, args.soft)
            elif args.task_type == 'gender':
                loss, _ = args.criterion(preds.detach(), genders, weight, args.soft)
                
            
        elif args.crt_type == 'ce':
            loss = args.criterion(preds, targets.max(1, keepdim=False)[1])
            
        if args.task_type == 'action':
            with torch.no_grad():
                sample_loss = args.ref_criterion(preds, targets.argmax(-1))
        else:
            with torch.no_grad():
                sample_loss = args.ref_criterion(preds, genders.argmax(-1))

        preds = F.softmax(preds, dim=-1)
        preds_max = preds.max(1, keepdim=True)[1]

        tensor = torch.tensor((), dtype=torch.float64)
        preds_exact = tensor.new_zeros(preds.size())
        for idx, item in enumerate(preds_max):
            preds_exact[idx, item] = 1
            
            

        res.append((image_ids, preds.detach().cpu(), targets.detach().cpu(), genders.detach().cpu(), preds_exact,\
                   weight.detach().cpu(), sample_loss.detach().cpu()))
        
        loss_logger.update(loss.item())
        
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)

        # Print log info

    # compute mean average precision score for verb classifier
    total_preds   = torch.cat([entry[1] for entry in res], 0).numpy()
    total_targets = torch.cat([entry[2] for entry in res], 0).numpy() if args.task_type == 'action' else torch.cat([entry[3] for entry in res], 0).numpy()
    total_genders = torch.cat([entry[3] for entry in res], 0).numpy()
    total_preds_exact = torch.cat([entry[4] for entry in res], 0).numpy()
    total_weight = torch.cat([entry[5] for entry in res], 0).numpy()
    total_loss = torch.cat([entry[6] for entry in res], 0).numpy()

    acc = np.mean(total_targets.argmax(-1) == total_preds_exact.argmax(-1))
    acc_male = np.mean(total_targets.argmax(-1)[total_genders[:, 0].nonzero()] == total_preds_exact.argmax(-1)[total_genders[:, 0].nonzero()])
    acc_female = np.mean(total_targets.argmax(-1)[total_genders[:, 1].nonzero()] == total_preds_exact.argmax(-1)[total_genders[:, 1].nonzero()])
    acc_diff = abs(acc_male - acc_female)
    
    if args.task_type == 'action' and args.dataset == 'imsitu':
        cnt = 0
        top5 = total_preds.argsort(-1)[:, -5:]
        match_t5 = np.zeros_like(total_targets.argmax(-1))

        for i, label in enumerate(total_targets.argmax(-1).reshape(-1,1)):
            if label in top5[i]:
                match_t5[i] = 1
        #         cnt += 1

        acc_t5 = np.mean(match_t5)
        acc_t5_male = np.mean(match_t5[total_genders[:, 0].nonzero()])
        acc_t5_female = np.mean(match_t5[total_genders[:, 1].nonzero()])
        acc_t5_diff = abs(acc_t5_female - acc_t5_male)

        cnt = 0
        top10 = total_preds.argsort(-1)[:, -10:]
        match_t10 = np.zeros_like(total_targets.argmax(-1))

        for i, label in enumerate(total_targets.argmax(-1).reshape(-1,1)):
            if label in top10[i]:
                match_t10[i] = 1
        #         cnt += 1

        # acc_t10 = cnt/(i + 1)
        acc_t10 = np.mean(match_t10)
        acc_t10_male = np.mean(match_t10[total_genders[:, 0].nonzero()])
        acc_t10_female = np.mean(match_t10[total_genders[:, 1].nonzero()])
        acc_t10_diff = abs(acc_t10_female - acc_t10_male)

    task_f1_score = f1_score(total_targets, total_preds_exact, average = 'macro')
    task_f1_score_male = f1_score(total_targets[total_genders[:, 0].nonzero()], total_preds_exact[total_genders[:, 0].nonzero()], average = 'macro')
    task_f1_score_female = f1_score(total_targets[total_genders[:, 1].nonzero()], total_preds_exact[total_genders[:, 1].nonzero()], average = 'macro')
    task_f1_score_diff = abs(task_f1_score_male - task_f1_score_female)

    man_idx = total_genders[:, 0].nonzero()[0]
    woman_idx = total_genders[:, 1].nonzero()[0]

    preds_man = total_preds[man_idx]
    preds_woman = total_preds[woman_idx]
    targets_man = total_targets[man_idx]
    targets_woman = total_targets[woman_idx]
    
    if args.task_type == 'action' and args.dataset == 'imsitu':
        dialog, diag_dict = gen_eval_diag(args.id2verb, total_preds_exact, total_targets, man_idx, woman_idx)
    
    none_idx = targets_woman.sum(0) == 0
    targets_woman = targets_woman[:, ~none_idx]
    preds_woman = preds_woman[:, ~none_idx]
    
    meanAP = average_precision_score(total_targets, total_preds, average='macro')
    meanAP_man = average_precision_score(targets_man, preds_man, average='macro')
    meanAP_woman = average_precision_score(targets_woman, preds_woman, average='macro')

#     print('loss', loss_logger.avg, epoch)
    args.logger_valid.info('######## Valid epoch  : {} ########, '.format(epoch))
    
    args.logger_valid.info('acc : {:.3f}'.format(acc))
    args.logger_valid.info('acc diff: {:.3f}'.format(acc_diff))
    
    if args.task_type == 'action' and args.dataset == 'imsitu':
        args.logger_valid.info('acc top5 : {:.3f}'.format(acc_t5))
        args.logger_valid.info('acc top5 diff : {:.3f}'.format(acc_t5_diff))
        args.logger_valid.info('acc top10 : {:.3f}'.format(acc_t10))
        args.logger_valid.info('acc top10 diff : {:.3f}'.format(acc_t10_diff))

    # print('loss : {:.3f}'.format(loss_logger.avg, epoch))
    args.logger_valid.info('task_f1_score : {:.3f}'.format(task_f1_score, ))
    args.logger_valid.info('task_f1_score diff: {:.3f}'.format(task_f1_score_diff, ))


    args.logger_valid.info('loss : {:.3f}'.format(loss_logger.avg, epoch))
    args.logger_valid.info('meanAP : {:.3f}'.format(meanAP, epoch))
    args.logger_valid.info('meanAP_man : {:.3f}'.format(meanAP_man, epoch))
    args.logger_valid.info('meanAP_woman : {:.3f}'.format(meanAP_woman, epoch))

    args.logger_valid.info('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    args.logger_valid.info('Valid epoch  : {}, meanAP: {:.2f}, meanAP_man: {:.2f}, meanAP_woman: {:.2f}'.format( \
        epoch, meanAP*100, meanAP_man*100, meanAP_woman*100))
    
    if args.task_type == 'action' and args.dataset == 'imsitu':
        args.logger_valid_verbose.info(dialog)
        args.valid_log_dict['dialog'].append(diag_dict)
        
        args.valid_log_dict['acc top5'].append(acc_t5)
        args.valid_log_dict['acc top5_male'].append(acc_t5_male)
        args.valid_log_dict['acc top5_female'].append(acc_t5_female)

        args.valid_log_dict['acc top10'].append(acc_t10)
        args.valid_log_dict['acc top10_male'].append(acc_t10_male)
        args.valid_log_dict['acc top10_female'].append(acc_t10_female)

        
    args.valid_log_dict['pred_score'].append(total_preds)
    args.valid_log_dict['target'].append(total_targets)
    args.valid_log_dict['gender'].append(total_genders)
    args.valid_log_dict['pred_exact'].append(total_preds_exact)
    args.valid_log_dict['weight'].append(total_weight)
    args.valid_log_dict['loss'].append(total_loss)
    
    
    args.valid_log_dict['acc'].append(acc)
    args.valid_log_dict['acc_male'].append(acc_male)
    args.valid_log_dict['acc_female'].append(acc_female)
    
    args.valid_log_dict['mAP'].append(meanAP)
    args.valid_log_dict['mAP_male'].append(meanAP_man)
    args.valid_log_dict['mAP_female'].append(meanAP_woman)
    
    args.valid_log_dict['fscore'].append(task_f1_score)
    args.valid_log_dict['fscore_male'].append(task_f1_score_male)
    args.valid_log_dict['fscore_female'].append(task_f1_score_female)
    
#         diag_dict['total_preds'] = total_preds
#         diag_dict['total_targets'] = total_targets
#         diag_dict['total_genders'] = total_genders
#         diag_dict['total_preds_exact'] = total_preds_exact    
    
#         with open(os.path.join(args.log_dir, 'val', 'eval_{}.pickle'.format(epoch)), 'wb') as handle:
#             pickle.dump(diag_dict, handle)

    return task_f1_score

            
def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.pth.tar'))
        
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
        
def setup_logger(logger_name, log_file, level=logging.INFO, verbose = True):
    l = logging.getLogger(logger_name)
    
    formatter = logging.Formatter('[%(asctime)s]  %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    if verbose:
        streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    if verbose:
        l.addHandler(streamHandler) 
    
    

def gen_eval_diag(id2verb, total_preds_exact, total_targets, man_idx, woman_idx):
    diag = str()
    diag_dict = dict()
    
    for (key, value) in id2verb.items():
        diag_dict[key] = {'verb':value}
        diag += '\n[ ' + str(key) + ' : ' + value + ' ] \t\t'

        true_idx = total_targets[:, key] == 1
        num_samples = sum(true_idx)

        TP = sum(total_preds_exact[true_idx][:, key])
        FP = sum(total_preds_exact[~true_idx][:, key])

        if (TP + FP) == 0:
            Pr = 0
        else:
            Pr = TP/(TP + FP)
        Rc = np.mean(total_targets[true_idx][:, key] == total_preds_exact[true_idx][:, key])

        diag += 'total ({}) : Prec : {:.2f}, Recall : {:.2f},  \t'.format(num_samples, Pr, Rc)
        diag_dict[key]['total'] = {'num':num_samples, 'Prec':Pr, 'Recall':Rc}

        true_idx = total_targets[man_idx][:, key] == 1
        num_samples = sum(true_idx)

        TP = sum(total_preds_exact[man_idx][true_idx][:, key])
        FP = sum(total_preds_exact[man_idx][~true_idx][:, key])
        
        if (TP + FP) == 0:
            Pr = 0
        else:
            Pr = TP/(TP + FP)
        Rc = np.mean(total_targets[man_idx][true_idx][:, key] == total_preds_exact[man_idx][true_idx][:, key])


        diag += 'man ({}) : Prec : {:.2f}, Recall : {:.2f},  \t'.format(num_samples, Pr, Rc)
        diag_dict[key]['man'] = {'num':num_samples, 'Prec':Pr, 'Recall':Rc}

        true_idx = total_targets[woman_idx][:, key] == 1
        num_samples = sum(true_idx)

        TP = sum(total_preds_exact[woman_idx][true_idx][:, key])
        FP = sum(total_preds_exact[woman_idx][~true_idx][:, key])

        if (TP + FP) == 0:
            Pr = 0
        else:
            Pr = TP/(TP + FP)
        Rc = np.mean(total_targets[woman_idx][true_idx][:, key] == total_preds_exact[woman_idx][true_idx][:, key])

        diag += 'woman ({}) : Prec : {:.2f}, Recall : {:.2f},  \n'.format(num_samples, Pr, Rc)
        diag_dict[key]['woman'] = {'num':num_samples, 'Prec':Pr, 'Recall':Rc}
        
    return diag, diag_dict
        
    return diag
        
        
if __name__ == "__main__":
    main()
    
    