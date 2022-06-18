import os
import sys
import errno
import torch
import random
import numbers
from math import nan
import numpy as np
from os import path

from sklearn.metrics import precision_score, f1_score, recall_score, average_precision_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

from torchvision.transforms import functional as F


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target)
        acc = correct.float().sum().mul_(1.0 / batch_size)
    return acc, pred


def calc_metrics(y_pred, y_true, y_scores):
    metrics = {}
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_true = torch.cat(y_true).cpu().numpy()
    try:
        y_scores = torch.cat(y_scores).cpu().numpy()
    except:
        y_scores = torch.cat(y_scores).cpu().detach().numpy()
    classes = unique_labels(y_true, y_pred)
    
    # precision score
    try:
        metrics['pr'] = precision_score(y_true, y_pred, average='weighted', zero_division = 0)
    except:
        metrics['pr'] = nan

    # recall score
    try:
        metrics['rec'] = recall_score(y_true, y_pred, average='weighted')
    except:
        metrics['rec'] = nan

    # f1 score
    try:
        f1_scores = f1_score(y_true, y_pred, average='weighted', labels=unique_labels(y_pred))
        metrics['f1'] = f1_scores.sum() / classes.shape[0]
    except:
        metrics['f1'] = nan
        
        
    Y = label_binarize(y_true, classes=classes.astype(int).tolist())    
    # AUC PR
    try:
        metrics['aucpr'] = average_precision_score(Y, y_scores, average='weighted')
    except:
        metrics['aucpr'] = nan

    # AUC ROC
    try:
        metrics['aucroc'] = roc_auc_score(Y, y_scores, average='weighted')
    except:
        metrics['aucroc'] = nan
        
    # Spec ROC
    try:
        metrics['aucroc'] = roc_auc_score(Y, y_scores, average='weighted')
    except:
        metrics['aucroc'] = nan
    
    #Confusion matrix
    try:
        metrics['confusion_mat'] = confusion_matrix(y_true, y_pred)
    except:
        metrics['confusion_mat'] = nan

    return metrics


def save_metrics(his_loss, his_acc, his_val_loss, his_val_acc, 
         his_pr, his_rec, his_f1, his_aucpr, his_aucroc,
         his_val_pr, his_val_rec, his_val_f1, his_val_aucpr, his_val_aucroc, 
         his_cm, his_val_cm,
         base_path_his):

    # Save
    np.save(path.join(base_path_his, "Loss"), np.array(his_loss))
    np.save(path.join(base_path_his, "Acc"), np.array(his_acc))
    np.save(path.join(base_path_his, "Pr"), np.array(his_pr))
    np.save(path.join(base_path_his, "Rec"), np.array(his_rec))
    np.save(path.join(base_path_his, "f1"), np.array(his_f1))
    np.save(path.join(base_path_his, "Aucpr"), np.array(his_aucpr))
    np.save(path.join(base_path_his, "Aucroc"), np.array(his_aucroc))
    np.save(path.join(base_path_his, "conf_mat"), np.array(his_cm))
    
    np.save(path.join(base_path_his, "Val_Loss"), np.array(his_val_loss))
    np.save(path.join(base_path_his, "Val_Acc"), np.array(his_val_acc))
    np.save(path.join(base_path_his, "Val_Pr"), np.array(his_val_pr))
    np.save(path.join(base_path_his, "Val_Rec"), np.array(his_val_rec))
    np.save(path.join(base_path_his, "Val_f1"), np.array(his_val_f1))
    np.save(path.join(base_path_his, "Val_Aucpr"), np.array(his_val_aucpr))
    np.save(path.join(base_path_his, "Val_Aucroc"), np.array(his_val_aucroc))
    np.save(path.join(base_path_his, "Val_conf_mat"), np.array(his_val_cm))
    
    
def save_results(y_pred, y_true, y_scores, base_path_his, mode):
     if not os.path.isdir(path.join(base_path_his, mode)):
        os.makedirs(path.join(base_path_his, mode))
        
     y_pred =  np.array(torch.cat(y_pred).cpu().numpy())
     y_true =  np.array(torch.cat(y_true).cpu().numpy())
     y_scores =  np.array(torch.cat(y_scores).cpu().numpy())

                       
     if(mode=="train"):
        np.save(path.join(base_path_his, "train", "preds"), y_pred)
        np.save(path.join(base_path_his, "train", "targets"), y_true)
        np.save(path.join(base_path_his, "train", "scores"), y_scores)
    
     elif(mode=="valid"):
        np.save(path.join(base_path_his, "valid", "preds"), y_pred)
        np.save(path.join(base_path_his, "valid", "targets"), y_true)
        np.save(path.join(base_path_his, "valid", "scores"), y_scores)
   
     elif(mode=="test"):
        np.save(path.join(base_path_his, "test", "preds"), y_pred)
        np.save(path.join(base_path_his, "test", "targets"), y_true)
        np.save(path.join(base_path_his, "test", "scores"), y_scores)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
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


class Logger(object):
    console = sys.stdout

    def __init__(self, fpath=None):
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class RandomFiveCrop(object):

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        # randomly return one of the five crops
        return F.five_crop(img, self.size)[random.randint(0, 4)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
