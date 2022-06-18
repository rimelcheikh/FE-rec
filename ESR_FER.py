"""
Experiments on FER+ published at AAAI-20 (Siqueira et al., 2020).

Reference:
    Siqueira, H., Magg, S. and Wermter, S., 2020. Efficient Facial Feature Learning with Wide Ensemble-based
    Convolutional Neural Networks. Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence
    (AAAI-20), pages 1Ã¢â‚¬â€œ1, New York, USA.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "1.0"


# External Libraries
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import argparse
import time
from datetime import timedelta
from sklearn.metrics import precision_score, f1_score, recall_score, average_precision_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from torchvision.transforms import ToTensor

# Standard Libraries
import os
from os import path, makedirs
import copy
from math import nan

# Modules
from utils import load_fer, umath
from ESR.esr_9_model import ESR
from utils_ESR import calc_metrics, save_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ferPlus_path', type=str, default='../DB/FER+/', help='Fer+ dataset path.')
    parser.add_argument('--bs', type=int, default=16, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay rate.')

    return parser.parse_args()


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x_base_to_process):
        x_base = F.relu(self.bn1(self.conv1(x_base_to_process)))
        x_base = self.pool(F.relu(self.bn2(self.conv2(x_base))))
        x_base = F.relu(self.bn3(self.conv3(x_base)))
        x_base = self.pool(F.relu(self.bn4(self.conv4(x_base))))

        return x_base


class Branch(nn.Module):
    def __init__(self):
        super(Branch, self).__init__()

        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, 8)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_branch_to_process):
        x_branch = F.relu(self.bn1(self.conv1(x_branch_to_process)))
        x_branch = self.pool(F.relu(self.bn2(self.conv2(x_branch))))
        x_branch = F.relu(self.bn3(self.conv3(x_branch)))
        x_branch = self.global_pool(F.relu(self.bn4(self.conv4(x_branch))))
        x_branch = x_branch.view(-1, 512)
        x_branch = self.fc(x_branch)

        return x_branch


class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()

        self.base = Base()
        self.branches = []

    def get_ensemble_size(self):
        return len(self.branches)

    def add_branch(self):
        self.branches.append(Branch())

    def forward(self, x):
        x_ensemble = self.base(x)

        y = []
        for branch in self.branches:
            y.append(branch(x_ensemble))

        return y

    @staticmethod
    def save(state_dicts, base_path_to_save_model, current_branch_save):
        if not path.isdir(path.join(base_path_to_save_model, str(len(state_dicts) - 1 - current_branch_save))):
            makedirs(path.join(base_path_to_save_model, str(len(state_dicts) - 1 - current_branch_save)))

        torch.save(state_dicts[0],
                   path.join(base_path_to_save_model,
                             str(len(state_dicts) - 1 - current_branch_save),
                             "Net-Base-Shared_Representations.pt"))

        for i in range(1, len(state_dicts)):
            torch.save(state_dicts[i],
                       path.join(base_path_to_save_model,
                                 str(len(state_dicts) - 1 - current_branch_save),
                                 "Net-Branch_{}.pt".format(i)))

        print("Network has been "
              "saved at: {}".format(path.join(base_path_to_save_model,
                                              str(len(state_dicts) - 1 - current_branch_save))))

    @staticmethod
    def load(device_to_load, ensemble_size):
        # Load ESR-9
        esr_9 = ESR(device_to_load)
        loaded_model = Ensemble()
        loaded_model.branches = []

        # Load the base of the network
        loaded_model.base = esr_9.base

        # Load branches
        for i in range(ensemble_size):
            loaded_model_branch = Branch()
            loaded_model_branch.conv1 = esr_9.convolutional_branches[i].conv1
            loaded_model_branch.conv2 = esr_9.convolutional_branches[i].conv2
            loaded_model_branch.conv3 = esr_9.convolutional_branches[i].conv3
            loaded_model_branch.conv4 = esr_9.convolutional_branches[i].conv4
            loaded_model_branch.bn1 = esr_9.convolutional_branches[i].bn1
            loaded_model_branch.bn2 = esr_9.convolutional_branches[i].bn2
            loaded_model_branch.bn3 = esr_9.convolutional_branches[i].bn3
            loaded_model_branch.bn4 = esr_9.convolutional_branches[i].bn4
            loaded_model_branch.fc = esr_9.convolutional_branches[i].fc
            loaded_model.branches.append(loaded_model_branch)

        return loaded_model

    def to_state_dict(self):
        state_dicts = [copy.deepcopy(self.base.state_dict())]

        for b in self.branches:
            state_dicts.append(copy.deepcopy(b.state_dict()))

        return state_dicts

    def to_device(self, device_to_process="cpu"):
        self.to(device_to_process)
        self.base.to(device_to_process)

        for b_td in self.branches:
            b_td.to(device_to_process)

    def reload(self, best_configuration):
        self.base.load_state_dict(best_configuration[0])

        for i in range(self.get_ensemble_size()):
            self.branches[i].load_state_dict(best_configuration[i + 1])


def evaluate(val_model_eval, val_loader_eval, val_criterion_eval, device_to_process="cpu", current_branch_on_training_val=0):
    running_val_loss = [0.0 for _ in range(val_model_eval.get_ensemble_size())]
    running_val_corrects = [0 for _ in range(val_model_eval.get_ensemble_size() + 1)]
    running_val_steps = [0 for _ in range(val_model_eval.get_ensemble_size())]
    y_pred_val, y_true_val, y_scores_val = [[] for _ in range(val_model_eval.get_ensemble_size() + 1)], [[] for _ in range(val_model_eval.get_ensemble_size() + 1)], [[] for _ in range(val_model_eval.get_ensemble_size() + 1)]

    for inputs_eval, labels_eval, _ in val_loader_eval:
        inputs_eval, labels_eval = inputs_eval.to(device_to_process), labels_eval.to(device_to_process)
        labels_eval = labels_eval.type(dtype=torch.long)
        outputs_eval = val_model_eval(inputs_eval)
        outputs_eval = outputs_eval[:val_model_eval.get_ensemble_size() - current_branch_on_training_val]

        # Ensemble prediction
        overall_preds = torch.zeros(outputs_eval[0].size()).to(device_to_process)

        for o_eval, outputs_per_branch_eval in enumerate(outputs_eval, 0):
            _, preds_eval = torch.max(outputs_per_branch_eval, 1)
            #preds_eval contains the index of the max in each element of the batch

            running_val_corrects[o_eval] += torch.sum(preds_eval == labels_eval).cpu().numpy()
            loss_eval = val_criterion_eval(outputs_per_branch_eval, labels_eval)
            running_val_loss[o_eval] += loss_eval.item()
            running_val_steps[o_eval] += 1
            y_pred_val[o_eval].append(preds_eval)
            y_true_val[o_eval].append(labels_eval)
            y_scores_val[o_eval].append(outputs_per_branch_eval.data)

            #table containing the number of predictions made for each element (in the batch) for each possible set of emotions
            for v_i, v_p in enumerate(preds_eval, 0):
                overall_preds[v_i, v_p] += 1
                 
        # Compute accuracy of ensemble predictions
        _, preds_eval = torch.max(overall_preds, 1)
        running_val_corrects[-1] += torch.sum(preds_eval == labels_eval).cpu().numpy()
        y_pred_val[-1].append(preds_eval)
        y_true_val[-1].append(labels_eval)
        y_scores_val[-1].append(outputs_eval[-1].data)

    for b_eval in range(val_model_eval.get_ensemble_size()):
        div = running_val_steps[b_eval] if running_val_steps[b_eval] != 0 else 1
        running_val_loss[b_eval] /= div
    

    return running_val_loss, running_val_corrects, y_pred_val, y_true_val, y_scores_val





base_path_experiment = "./experiments/ESR/FER/"
name_experiment = "exp_2"


def main():
    args = parse_args()
    #Experimental variables
    
    num_folds = 5

    # Make dir
    if not path.isdir(path.join(base_path_experiment, name_experiment)):
        makedirs(path.join(base_path_experiment, name_experiment))

    # Define transforms
    data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomAffine(degrees=30,
                                               translate=(.1, .1),
                                               scale=(1.0, 1.25),
                                               interpolation=InterpolationMode.BILINEAR)]
    

    # Running device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(0, num_folds):
        print("Starting: {}".format(str(name_experiment)))
        print("K-fold Cross-validation: {}--{}".format(fold + 1, num_folds))
        print("Running on {}".format(device))
        
        num_branches_trained_network = 9
        validation_interval = 1
        current_branch_on_training = 8
    
        # Load network trained on AffectNet
        net = Ensemble.load(device, num_branches_trained_network)
    
        # Send params to device
        net.to_device(device)
    
        # Set optimizer
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam([{"params": net.base.parameters(), "weight_decay": args.wd},
                                         {"params": net.branches[0].parameters(), "weight_decay": args.wd}])
            for b in range(1, net.get_ensemble_size()):
                optimizer.add_param_group({"params": net.branches[b].parameters(), "weight_decay": args.wd})
    
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([{"params": net.base.parameters(), "lr": args.lr, "momentum": args.momentum,"weight_decay": args.wd},
                                         {"params": net.branches[0].parameters(),"lr": args.lr, "momentum": args.momentum,"weight_decay": args.wd}])
            for b in range(1, net.get_ensemble_size()):
                optimizer.add_param_group({"params": net.branches[b].parameters(), "lr": args.lr, "momentum": args.momentum,"weight_decay": args.wd})
    
        else:
            raise ValueError("Optimizer not supported.")
        
        # Define criterion
        criterion = nn.CrossEntropyLoss()
        
        # Load validation set
        # max_loaded_images_per_label=100000 loads the whole validation set
        val_data = load_fer.FERplus(fold, idx_set=1,
                                 max_loaded_images_per_label=100000,
                                 transforms=None,
                                 base_path_to_FER_plus=args.ferPlus_path)
        val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory = True)
        
        test_data = load_fer.FERplus(fold, idx_set=2,
                                 max_loaded_images_per_label=1000000,
                                 transforms=None,
                                 base_path_to_FER_plus=args.ferPlus_path)
        test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory = True)
        
        # Load training data
        train_data = load_fer.FERplus(fold, idx_set=0,
                                   max_loaded_images_per_label=1000000,
                                   transforms=transforms.Compose(data_transforms),
                                   base_path_to_FER_plus=args.ferPlus_path)
        train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory = True)
        # Fine-tune ESR-9
        for branch_on_training in range(num_branches_trained_network):                            
            torch.cuda.empty_cache()
    
            # Best network
            best_ensemble = net.to_state_dict()
            best_ensemble_acc = 0.0
            best_ensemble_rec = 0.0
    
            # Initialize ptint
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
    
            
    
            # Training branch
            for epoch in range(args.epochs):
                start = time.time()
    
    
                running_loss = 0.0
                running_corrects = [0.0 for _ in range(net.get_ensemble_size())]
                running_updates = 0
                
                y_pred, y_true, y_scores = [[] for _ in range(net.get_ensemble_size())], [[] for _ in range(net.get_ensemble_size())], [[] for _ in range(net.get_ensemble_size())]
                y_pred_val, y_true_val, y_scores_val = [[] for _ in range(net.get_ensemble_size()+1)],[[] for _ in range(net.get_ensemble_size()+1)],[[] for _ in range(net.get_ensemble_size()+1)]
    
                #scheduler.step()
    
                for inputs, labels, _ in train_loader:
                    # Get the inputs
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.type(dtype=torch.long)
    
                    # Set gradients to zero
                    optimizer.zero_grad()
    
                    # Forward
                    outputs = net(inputs)
                    confs_preds = [torch.max(o, 1) for o in outputs]
    
                    # Compute loss
                    loss = 0.0
                    for i_4 in range(net.get_ensemble_size() - current_branch_on_training):
                        preds = confs_preds[i_4][1]
                        running_corrects[i_4] += torch.sum(preds == labels).cpu().numpy()
                        loss += criterion(outputs[i_4], labels)
                        
                        # collect for metrics
                        y_pred[i_4].append(preds)
                        y_true[i_4].append(labels)
                        y_scores[i_4].append(outputs[i_4].data)
    
                    # Backward
                    loss.backward()
    
                    # Optimize
                    optimizer.step()
    
                    # Save loss
                    running_loss += loss.item()
                    running_updates += 1
                
                scheduler.step()
                
                save_results(y_pred, y_true, y_scores, os.path.join(base_path_experiment, name_experiment, 'Fold-{:02d}'.format(fold + 1), "Branch_"+str(branch_on_training+1), "epoch_"+str(epoch+1)), "train")
    
    
                
                
    
                    
        
                # Statistics
                print("[Branch {:d}, Epochs {:d}--{:d}] "
                      "Loss: {:.4f} Acc: {}".format(net.get_ensemble_size() - current_branch_on_training,
                                                    epoch + 1,
                                                    args.epochs,
                                                    running_loss / running_updates,
                                                    np.array(running_corrects) / len(train_data)))
                
                # Validation
                torch.cuda.empty_cache()
                if ((epoch % validation_interval) == 0) or ((epoch + 1) == args.epochs):
                    net.eval()
    
                    val_loss, val_corrects, y_pred_val, y_true_val, y_scores_val = evaluate(net, val_loader, criterion, device, current_branch_on_training)
                    
                    
                    print("\nValidation - [Branch {:d}, Epochs {:d}--{:d}] Loss: {:.4f} Acc: {}\n\n".format(net.get_ensemble_size() - current_branch_on_training,
                                                                                                            epoch + 1,
                                                                                                            args.epochs,
                                                                                                            val_loss[branch_on_training],
                                                                                                            np.array(val_corrects) / len(val_data)))
    
    
                    
                    if not path.isdir(path.join(base_path_experiment, name_experiment, "models")):
                        makedirs(path.join(base_path_experiment, name_experiment, "models"))
    
                    # Save best ensemble
                    ensemble_acc = (float(val_corrects[-1]) / len(val_data))
                    if ensemble_acc >= best_ensemble_acc:
                        best_ensemble_acc = ensemble_acc
                        best_ensemble = net.to_state_dict()
                        save_checkpoint(epoch, net ,base_path_experiment+name_experiment+"/models", weights_tag='best_weights.pth', model_tag='best_model.pth')
    
                    save_results(y_pred_val, y_true_val, y_scores_val, os.path.join(base_path_experiment, name_experiment, 'Fold-{:02d}'.format(fold + 1), "Branch_"+str(branch_on_training+1), "epoch_"+str(epoch+1)), "valid")
     
                    test(epoch, test_loader, net, fold, current_branch_on_training)            
     
        
                    net.train()
                end = time.time()
                print(f'[*] epoch time = {end - start:.2f}s | ')
                if not os.path.isdir(path.join(base_path_experiment, name_experiment, "time")):
                     os.makedirs(path.join(base_path_experiment, name_experiment, "time"))
                np.save(path.join(base_path_experiment, name_experiment, "time", "branch_"+str(net.get_ensemble_size() - current_branch_on_training)), end - start)
                
                
            # Change branch on training
            if current_branch_on_training > 0:
    
                # Reload best configuration
                net.reload(best_ensemble)
    
                # Set optimizer
                if args.optimizer == 'adam':
                    optimizer = torch.optim.Adam([{"params": net.base.parameters(), "weight_decay": 1e-4},
                                                 {"params": net.branches[0].parameters(),"weight_decay": 1e-4}])
                    for b in range(1,net.get_ensemble_size()):
                        if b != (net.get_ensemble_size() - current_branch_on_training):
                            optimizer.add_param_group({"params": net.branches[b].parameters(), "weight_decay": 1e-4})
                
                elif args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD([{"params": net.base.parameters(), "lr": args.lr, "momentum": args.momentum,"weight_decay": 1e-4},
                                                 {"params": net.branches[0].parameters(),"lr": args.lr, "momentum": args.momentum,"weight_decay": 1e-4}])
                    for b in range(1,net.get_ensemble_size()):
                        if b != (net.get_ensemble_size() - current_branch_on_training):
                            optimizer.add_param_group({"params": net.branches[b].parameters(), "lr": 0.02, "momentum": 0.9})
                else:
                    raise ValueError("Optimizer not supported.")
        
                
                # Change branch on training
                current_branch_on_training -= 1
                
            # Finish training after fine-tuning all branches
            else:
                break



def test(epoch, test_loader, model, fold, current_branch_on_training=0):

    y_pred, y_true, y_scores = [[] for _ in range(model.get_ensemble_size() + 1)], [[] for _ in range(model.get_ensemble_size() + 1)], [[] for _ in range(model.get_ensemble_size() + 1)]

    for inputs, labels, _ in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.type(dtype=torch.long)
        outputs = model(inputs)
        outputs = outputs[:model.get_ensemble_size() - current_branch_on_training]

        # Ensemble prediction
        overall_preds = torch.zeros(outputs[0].size()).to(device)

        for o, outputs_per_branch in enumerate(outputs, 0):
            _, preds = torch.max(outputs_per_branch, 1)
            #preds_eval contains the index of the max in each element of the batch

            y_pred[o].append(preds)
            y_true[o].append(labels)
            y_scores[o].append(outputs_per_branch.data)

            #table containing the number of predictions made for each element (in the batch) for each possible set of emotions
            for v_i, v_p in enumerate(preds, 0):
                overall_preds[v_i, v_p] += 1
                 
        # Compute accuracy of ensemble predictions
        _, preds = torch.max(overall_preds, 1)
        
        y_pred[-1].append(preds)
        y_true[-1].append(labels)
        y_scores[-1].append(outputs[-1].data)
    


            
    save_results(y_pred, y_true, y_scores, os.path.join(base_path_experiment, name_experiment, 'Fold-{:02d}'.format(fold + 1), "Branch_"+str(model.get_ensemble_size() - current_branch_on_training), "epoch_"+str(epoch+1)), "test")




def save_checkpoint(epoch, model, path, weights_tag, model_tag):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, weights_tag))
    
    torch.save(model, os.path.join(path, model_tag))

if __name__ == "__main__":
    print("Processing...")
    start = time.time()
    main()
    end = time.time()
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    print("Process has finished!")
