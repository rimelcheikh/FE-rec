import numpy as np
import torch
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, recall_score, average_precision_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import os
import sklearn


from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


    
    
def accuracy(pred, target):
    acc = accuracy_score(target, pred)
    return acc

def balanced_accuracy(pred, target):
    bal_acc = balanced_accuracy_score(target, pred)
    return bal_acc


def precision(pred, target):          
    pr = {}
    for avg in [None, 'micro', 'macro', 'weighted']:
        pr[str(avg)] = precision_score(target, pred, average=avg, zero_division=0)
    return pr

def recall(pred, target):     #tp / (tp + fn)
    rec = {}
    for avg in [None, 'micro', 'macro', 'weighted']:
        rec[str(avg)] = recall_score(target, pred, average=avg, zero_division=0)
    return rec

def f1score(pred, target):
    f1 = {}
    for avg in [None, 'micro', 'macro', 'weighted']:
        f1[str(avg)] = f1_score(target, pred, average=avg, zero_division=0)
    return f1
    
def aucroc(pred, scores, target):
    classes = unique_labels(target, pred)
    target = label_binarize(target, classes=classes.astype(int).tolist())  
    aucrc = {}
    for strat in ['ovo', 'ovr']:
        for avg in [None, 'micro', 'macro', 'weighted']:
            aucrc[str(avg)] = roc_auc_score(target, scores, average=avg, multi_class=strat)
    return aucrc
    

def conf_mat(pred, target):
    mat = confusion_matrix(target, pred,normalize='true')
    return mat



if __name__ == '__main__':

    
    for ds in ["an"]:
        print("----------------------",ds,"----------------------")
        
        for net in ["esr",'scn','dacl']:
            print("###############",net,"###############")
            
            best_acc = 0.0
            acc, bal_acc, rec, pr, f1, aucrc, mat = [], [], [], [], [], [], []
            
            n_f = 5
            
            if ds == "fer":
                exp = "exp_1"
                
            elif ds == "ck":
                exp = "exp_2"
            
            elif ds == "an" and net=="esr":
                exp = "exp_1"
                n_f = 2
                
            elif ds == "an" and net=="scn":
                exp = "exp_1"
                
            elif ds == "an" and net=="dacl":
                exp = "exp_2"
                
            
            path = "./experiments/"+net+"/"+ds+"/"+exp
            
            for f in range(n_f):
                
                fold = "Fold-0"+str(f+1)
                path = "./experiments/"+net+"/"+ds+"/"+exp+"/"+fold
            
                #look for epoch that gives best test_acc
                mode = 'test'
                best_acc = 0.0
                
                if(net != "esr"):
                    for epoch in os.listdir(path):
                        if(epoch != "models" and epoch != "time"):
                            results_path = path + "/"+epoch+"/"+mode
                            preds = np.load(results_path+"/preds.npy", allow_pickle=True)
                            targets = np.load(results_path+"/targets.npy", allow_pickle=True)
            
                            if accuracy(preds, targets) > best_acc:
                                best_acc = accuracy(preds, targets)
                                best_epoch = epoch

                else:
                    if ds == 'ck':
                        branch = "Branch_4"
                        for epoch in os.listdir(path+'/'+branch):
                            results_path = path +'/'+branch + "/"+epoch+"/"+mode
                            preds = np.load(results_path+"/preds.npy", allow_pickle=True).astype(int)[-1]
                            targets = np.load(results_path+"/targets.npy", allow_pickle=True).astype(int)[-1]
                            
                            if accuracy(preds, targets) > best_acc:
                                best_acc = accuracy(preds, targets)
                                best_epoch = epoch
                        
                    elif ds == 'fer':
                        branch = "Branch_9"
                        for epoch in os.listdir(path+'/'+branch):
                            results_path = path +'/'+branch + "/"+epoch+"/"+mode
                            preds = np.load(results_path+"/preds.npy", allow_pickle=True)[-1].astype(int)
                            targets = np.load(results_path+"/targets.npy", allow_pickle=True)[-1].astype(int)
                            
                            if accuracy(preds, targets) > best_acc:
                                best_acc = accuracy(preds, targets)
                                best_epoch = epoch
                            
                    elif ds == 'an':
                        branch = "Branch_9"
            
                        for epoch in os.listdir(path+'/'+branch):
                            results_path = path +'/'+branch + "/"+epoch+"/"+mode
                            preds = np.load(results_path+"/preds.npy", allow_pickle=True).astype(int)[-1]
                            targets = np.load(results_path+"/targets.npy", allow_pickle=True).astype(int)[-1]
                            
                            if accuracy(preds, targets) > best_acc:
                                best_acc = accuracy(preds, targets)
                                best_epoch = epoch
            
                
                
                
                for mode in ['test']:
                    if(net != "esr"):
                        results_path = path + "/"+best_epoch+"/"+mode
                        
                        preds = np.load(results_path+"/preds.npy", allow_pickle=True)
                        targets = np.load(results_path+"/targets.npy", allow_pickle=True)
                        scores = np.load(results_path+"/scores.npy", allow_pickle=True)     
                
                    else:
                        if ds == 'ck':
                            branch = "Branch_4"
                        else:
                            branch = "Branch_9"
            
                        results_path = path +'/'+branch + "/"+epoch+"/"+mode
                
                        preds = np.load(results_path+"/preds.npy", allow_pickle=True).astype(int)[-1]
                        targets = np.load(results_path+"/targets.npy", allow_pickle=True).astype(int)[-1]
                        scores = np.load(results_path+"/scores.npy", allow_pickle=True).astype(int)[-1]
                        
                    """if ds == 'ck':
                        m = preds
                        for hh in range(len(m)):
                            if(m[hh] == 0): m[hh] = 6
                            elif(m[hh] == 1): m[hh] = 5
                            elif(m[hh] == 2): m[hh] = 4
                            elif(m[hh] == 3): m[hh] = 1
                            elif(m[hh] == 4): m[hh] = 2
                            elif(m[hh] == 5): m[hh] = 3
                            elif(m[hh] == 6): m[hh] = 7
                            elif(m[hh] == 7): m[hh] = 0
                        preds = m
                        
                        m = targets
                        for hh in range(len(m)):
                            if(m[hh] == 0): m[hh] = 6
                            elif(m[hh] == 1): m[hh] = 5
                            elif(m[hh] == 2): m[hh] = 4
                            elif(m[hh] == 3): m[hh] = 1
                            elif(m[hh] == 4): m[hh] = 2
                            elif(m[hh] == 5): m[hh] = 3
                            elif(m[hh] == 6): m[hh] = 7
                            elif(m[hh] == 7): m[hh] = 0
                        targets = m
                        
                        m = scores
                        for hh in range(len(m)):
                            m[hh][0] = scores[hh][6]
                            m[hh][1] = scores[hh][5]
                            m[hh][2] = scores[hh][4]
                            m[hh][3] = scores[hh][1]
                            m[hh][4] = scores[hh][2]
                            m[hh][5] = scores[hh][3]
                            m[hh][6] = scores[hh][7]
                            m[hh][7] = scores[hh][0]
                        scores = m"""
            
                    for avrg in ["weighted"]:
                        acc.append(accuracy(preds, targets))
                        bal_acc.append(balanced_accuracy(preds, targets))
                        rec.append(recall(preds, targets)[avrg])
                        pr.append(precision(preds, targets)[avrg])
                        #f1.append(f1score(preds, targets)[avrg])
                        aucrc.append(aucroc(preds, scores, targets)[avrg])
                        mat.append(conf_mat(preds, targets))
                        f1.append((2*recall(preds, targets)[avrg]*precision(preds, targets)[avrg])/(precision(preds, targets)[avrg]+recall(preds, targets)[avrg]))
                           
            
            print(best_epoch, avrg, ':')
            print('acc :    ', '%.3f'%(np.mean(acc)), '±',  '%.3f'%(np.std(acc)))
            print('bal acc:', '%.3f'%(np.mean(bal_acc)), '±',  '%.3f'%(np.std(bal_acc)))
            """print('pr :    ', '%.3f'%(np.mean(pr)), '±',  '%.3f'%(np.std(pr)))
            print('rec :    ', '%.3f'%(np.mean(rec)), '±',  '%.3f'%(np.std(rec)))
            print('f1 :    ', '%.3f'%(np.mean(f1)), '±',  '%.3f'%(np.std(f1)))"""
            #print('aucroc :    ', '%.3f'%(np.mean(aucrc)), '±',  '%.3f'%(np.std(aucrc)))
            
            
            
            """mat_all = np.zeros((8,8))
            
            for ff in range(n_f):
                x = np.array(mat[ff])
                mat_all += x
            
            #if normalize='true'
            #print(((mat_all/n_f)*100).round(1))
            #mat_all = ((mat_all/n_f)*100).round(1)
            
            #if normalize='false'
            #print(((mat_all/n_f)).round(0))
            #mat_all = ((mat_all/n_f)).round(0)
            
            labels = ['Ne', 'Ha', 'Sa', 'Su','Fe', 'Di', 'An', 'Co']
            
                
            #a = sns.heatmap(mat_all, annot=True,fmt=".4g",cmap='Blues').plot()

            
            a = ConfusionMatrixDisplay(mat_all,display_labels=labels).plot( values_format='.4g')#.show()
            plt.savefig("./confusion_matrices"+net+"_"+ds+".png")
            plt.show()"""
                
                
                
        
        print("-------------------------------------------------------------------------")   
    print("#########################################################################")        
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    