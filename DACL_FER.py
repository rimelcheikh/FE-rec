import argparse
import random
import pprint
import time
import sys
import os
import numpy as np
from os import makedirs

from datetime import timedelta
from workspace import Workspace

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models_DACL.resnet_FER import resnet18
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils_DACL import save_results, Logger, AverageMeter, accuracy, calc_metrics, RandomFiveCrop, save_metrics
from utils import load_fer
from tqdm import tqdm

# centerloss module
from loss import SparseCenterLoss

parser = argparse.ArgumentParser(description='DACL for FER in the wild')
parser.add_argument('--arch', type=str, default="resnet18")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--lamb', type=float, default=0.01)
parser.add_argument('--pretrained', type=str, default='msceleb')
parser.add_argument('--deterministic', default=False, action='store_true')

base_path_experiment = "./experiments/DACL/FER/"
name_experiment = "exp_xx"
def main(cfg):
    num_folds = 5
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if cfg['deterministic']:
        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        cudnn.deterministic = True
        cudnn.benchmark = False
        
    # Make dir
    if not os.path.isdir(os.path.join(base_path_experiment, name_experiment)):
        makedirs(os.path.join(base_path_experiment, name_experiment))

    # Loading RAF-DB
    # -----------------

    for fold in range(0, num_folds):
        print("K-fold Cross-validation: {}--{}".format(fold + 1, num_folds))
        print('[>] Loading dataset '.ljust(64, '-'))

        data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomAffine(degrees=30,
                                                    translate=(.1, .1),
                                                    scale=(1.0, 1.25),
                                                    interpolation=transforms.InterpolationMode.BILINEAR)]
        #training set
        train_data = load_fer.FERplus(fold, idx_set=0,
                                       max_loaded_images_per_label=200,
                                       transforms=transforms.Compose(data_transforms),
                                       base_path_to_FER_plus='C:/Users/rielcheikh/Desktop/FER/DB/FER+/')
    
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory = True)
    
        # validation set
        val_data = load_fer.FERplus(fold, idx_set=1,
                                 max_loaded_images_per_label=200,
                                 transforms=None,
                                 base_path_to_FER_plus='C:/Users/rielcheikh/Desktop/FER/DB/FER+/')
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory = True)
        
        
        #testing set
        test_data = load_fer.FERplus(fold, idx_set=2,
                                            max_loaded_images_per_label=200,
                                            transforms=None,
                                            base_path_to_FER_plus='C:/Users/rielcheikh/Desktop/FER/DB/FER+/')
    
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2, pin_memory = True)
    
        
        print('[*] Loaded dataset!')
    
        # Create Model
        # ------------
        print('[>] Model '.ljust(64, '-'))
        if cfg['arch'] == 'resnet18':
            feat_size = 512
            if not cfg['pretrained'] == '':
                model = resnet18(pretrained=cfg['pretrained'])
                model.fc = nn.Linear(feat_size, 8)
            else:
                print('[!] model is trained from scratch!')
                model = resnet18(num_classes=8, pretrained=cfg['pretrained'])
        else:
            raise NotImplementedError('only working with "resnet18" now! check cfg["arch"]')
        model = torch.nn.DataParallel(model).to(device)
        print('[*] Model initialized!')
    
        # define loss function (criterion) and optimizer
        # ----------------------------------------------
        criterion = {
            'softmax': nn.CrossEntropyLoss().to(device),
            'center': SparseCenterLoss(8, feat_size).to(device)
        }
        optimizer = {
            'softmax': torch.optim.Adam(params= model.parameters(), weight_decay= 1e-4),
            'center': torch.optim.Adam(params= model.parameters(), weight_decay= 1e-4)
        }
        # lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer['softmax'], step_size=10, gamma=0.5, last_epoch=-1)
        
        # training and evaluation
        # -----------------------
        global best_valid
        best_valid = dict.fromkeys(['acc', 'rec', 'f1', 'aucpr', 'aucroc'], 0.0)
    
        print('[>] Begin Training '.ljust(64, '-'))
        for epoch in range(1, cfg['epochs'] + 1):
    
            print(list(model.children())[0])
            start = time.time()
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, cfg, fold)
            # validate for one epoch
            validate(val_loader, model, criterion, epoch, cfg, fold)            
            
            test(epoch, test_loader, model, fold)
                 
    
            # progress
            end = time.time()
            progress = (
                f'[*] epoch time = {end - start:.2f}s | '
                f'lr = {optimizer["softmax"].param_groups[0]["lr"]}\n'
            )
            print(progress)
    
            # lr step
            scheduler.step()
            
            
          
    
        # best valid info
        # ---------------
        print('[>] Best Valid '.ljust(64, '-'))
        stat = (
            f'[+] acc={best_valid["acc"]:.4f}\n'
            f'[+] rec={best_valid["rec"]:.4f}\n'
            f'[+] f1={best_valid["f1"]:.4f}\n'
            f'[+] aucpr={best_valid["aucpr"]:.4f}\n'
            f'[+] aucroc={best_valid["aucroc"]:.4f}'
        )
        print(stat)


def train(train_loader, model, criterion, optimizer, epoch, cfg, fold):
    losses = {
        'softmax': AverageMeter(),
        'center': AverageMeter(),
        'total': AverageMeter()
    }
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []
    global running_loss, running_acc

    # switch to train mode
    model.train()

    with tqdm(total=int(len(train_loader.dataset) / cfg['batch_size'])) as pbar:
        for i, (images, target, _) in enumerate(train_loader):
            images = images.to(device)
            target = torch.tensor(target, dtype=torch.long)
            target = target.to(device)

            # compute output
            feat, output, A = model(images)

            l_softmax = criterion['softmax'](output, target)
            l_center = criterion['center'](feat, A, target)
            l_total = l_softmax + cfg['lamb'] * l_center

            # measure accuracy and record loss
            acc, pred = accuracy(output, target)
            losses['softmax'].update(l_softmax.item(), images.size(0))
            losses['center'].update(l_center.item(), images.size(0))
            losses['total'].update(l_total.item(), images.size(0))
            accs.update(acc.item(), images.size(0))
            
            running_loss = losses['total'].avg
            running_acc = accs.avg

            # collect for metrics
            y_pred.append(pred)
            y_true.append(target)
            y_scores.append(output.data)

            # compute grads + opt step
            optimizer['softmax'].zero_grad()
            optimizer['center'].zero_grad()
            l_total.backward()
            optimizer['softmax'].step()
            optimizer['center'].step()

            # progressbar
            pbar.set_description(f'TRAINING [{epoch:03d}/{cfg["epochs"]}]')
            pbar.set_postfix({'L': losses["total"].avg,
                              'Ls': losses["softmax"].avg,
                              'Lsc': losses["center"].avg,
                              'acc': accs.avg})
            pbar.update(1)

    progress = (
        f'[-] TRAIN [{epoch:03d}/{cfg["epochs"]}] | '
        f'L={losses["total"].avg:.4f} | '
        f'acc={accs.avg:.4f} | '
    )
    print(progress)
    
    save_results(y_pred, y_true, y_scores, os.path.join(base_path_experiment, name_experiment, 'Fold-{:02d}'.format(fold + 1), "epoch_"+str(epoch)), "train")

    


def validate(valid_loader, model, criterion, epoch, cfg, fold):
    losses = {
        'softmax': AverageMeter(),
        'center': AverageMeter(),
        'total': AverageMeter()
    }
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []
    global running_val_loss, running_val_acc

    # switch to evaluate mode
    model.eval()

    with tqdm(total=int(len(valid_loader.dataset) / cfg['batch_size'])) as pbar:
        with torch.no_grad():
            for i, (images, target, _) in enumerate(valid_loader):

                images = images.to(device)
                #target = torch.tensor(target, dtype=torch.long)
                target = target.to(device)

                # compute output
                feat, output, A = model(images)
                l_softmax = criterion['softmax'](output, target)
                l_center = criterion['center'](feat, A, target)
                l_total = l_softmax + cfg['lamb'] * l_center

                # measure accuracy and record loss
                acc, pred = accuracy(output, target)
                losses['softmax'].update(l_softmax.item(), images.size(0))
                losses['center'].update(l_center.item(), images.size(0))
                losses['total'].update(l_total.item(), images.size(0))
                accs.update(acc.item(), images.size(0))

                running_val_loss = losses['total'].avg
                running_val_acc = accs.avg
                # collect for metrics
                y_pred.append(pred)
                y_true.append(target)
                y_scores.append(output.data)

                # progressbar
                pbar.set_description(f'VALIDATING [{epoch:03d}/{cfg["epochs"]}]')
                pbar.update(1)

    progress = (
        f'[-] VALID [{epoch:03d}/{cfg["epochs"]}] | '
        f'L={losses["total"].avg:.4f} | '
        f'acc={accs.avg:.4f} | '

    )
    print(progress)

    if not os.path.isdir(os.path.join(base_path_experiment,name_experiment,"models")):
        os.makedirs(os.path.join(base_path_experiment,name_experiment,"models"))
                
             
    if accs.avg > best_valid['acc']:        
        save_checkpoint(epoch, model,base_path_experiment+name_experiment+"/models", weights_tag='best_weights.pth', model_tag='best_model.pth')

    save_results(y_pred, y_true, y_scores, os.path.join(base_path_experiment, name_experiment, 'Fold-{:02d}'.format(fold + 1), "epoch_"+str(epoch)), "valid")
        
    best_valid['acc'] = max(best_valid['acc'], accs.avg)
    
    


def test(epoch, test_loader, model, fold):
    y_pred, y_true, y_scores = [], [], []
    # switch to evaluate mode
    model.eval()
    print("ààààààààààààààààààààààààààààà")
    with torch.no_grad():
        for i, (images, target, _) in enumerate(test_loader):
                
            images = images.to(device)
            target = target.to(device)
            
            
            """layers = list(list(model.children())[0].children())

            x = layers[0](images)
            
            conv_layer_output = None
            for i, layer in enumerate(layers[1:8]+layers[10:12]):
                x = layer(x)
                print(i,layer)
                print(x.shape)
                print("--------------------------")
                
                if i == 6: ## Output after 3rd Convolution layer
                    for i_last, layer_last in enumerate(list(layer.children())[1].children()):
                        if i_last == 3:
                            conv_layer_output = x  """

            # compute output and prediction
            _, output, _ = model(images)
            pred = torch.argmax(output, dim=1)
            
            y_pred.append(pred)
            y_true.append(target)
            y_scores.append(output.data)
            
        save_results(y_pred, y_true, y_scores, os.path.join(base_path_experiment, name_experiment, 'Fold-{:02d}'.format(fold + 1), "epoch_"+str(epoch)), "test")
    #print("ààààààààààààààààààààààààààààà")


def save_checkpoint(epoch, model, path, weights_tag, model_tag):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, weights_tag))
    
    torch.save(model, os.path.join(path, model_tag))


if __name__ == '__main__':

    # setting up workspace
    args = parser.parse_args()
    workspace = Workspace(args)
    cfg = workspace.config


    # -----------------
    start = time.time()
    main(cfg)
    end = time.time()
    # -----------------

    print('\n[*] Fini! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()