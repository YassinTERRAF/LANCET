import os
import pickle
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import models
import data_loader
from pytorch_model_summary import summary
import argparse
from torch.autograd import Variable
from sklearn.metrics import classification_report
from sklearn import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
import random

# Define emotion categories
target_names = ['neutral', 'happy', 'sad', 'angry']

def parse_args():
    parser = argparse.ArgumentParser(description='...')
    # features
    parser.add_argument('-f', '--features_to_use',default='mfcc',type=str,help='{"mfcc" , "logfbank","fbank","spectrogram","melspectrogram"}')
    parser.add_argument('-i', '--impro_or_script',default='impro',type=str,help='select features')
    parser.add_argument('-s', '--sample_rate',default=16000,type=int,help='sample rate, default is 16000')
    parser.add_argument('-n', '--nmfcc',default=26,type=int,help='MFCC coefficients')
    parser.add_argument('--train_overlap',default=1.6,type=float,help='train dataset overlap')
    parser.add_argument('--test_overlap',default=1.6,type=float,help='test dataset overlap')
    parser.add_argument('--segment_length',default=1.8,type=float,help='segment length')
    parser.add_argument('--toSaveFeatures',default=False,type=bool,help='Save features')
    parser.add_argument('--loadFeatures',default=True,type=bool,help='load features')
    parser.add_argument('--featuresFileName',default=None,type=str,help='features file name')
    # model
    parser.add_argument('-m', '--model',default='LANCET',type=str,help='specify models')
    parser.add_argument('--head',default=4,type=int,help='head numbers')
    parser.add_argument('--attn_hidden',default=64,type=int,help='attention hidden size')
    parser.add_argument('--SaveModel',default=True,type=bool,help='Save model')
    # Datasets
    parser.add_argument('--split_rate',default=0.8,type=float,help='dataset split rate')
    parser.add_argument('--aug',default=None,type=str,help='augmentation')
    parser.add_argument('--padding',default=None,type=str,help='padding')
    # training
    parser.add_argument('--seed',default=987654,type=int,help='random seed')
    parser.add_argument('-b','--batch_size',default=32,type=int,help='batch size')
    parser.add_argument('-l','--learning_rate',default=0.001,type=float,help='learning rate')
    parser.add_argument('--lr_min',default=1e-6,type=float,help='minimum lr')
    parser.add_argument('--lr_schedule',default='exp',type=str,help='lr schedule')
    parser.add_argument('--optimizer',default='adam',type=str,help='optimizer')
    parser.add_argument('-e','--epochs',default=50,type=int,help='epochs')
    parser.add_argument('--iter',default=1,type=int,help='iterations')
    parser.add_argument('-g','--gpu',default=None,type=int,help='specify gpu device')
    parser.add_argument('--weight',default=None,type=bool)
    # parser.add_argument('--mixup',default=None,type=bool)
    parser.add_argument('--alpha',default=0.5,type=float, help='mixing up trick, using beta distribution')
    # Config file
    parser.add_argument('-c', '--config', default=None, type=str, help='models')
    parser.add_argument('-d', '--datadir', default='', type=str, help='Directory where features are stored')
    parser.add_argument('--resultdir', default='', type=str, help='Directory to save results')
    parser.add_argument('--seeds_file', default='', type=str, help='File containing seeds')

    args = parser.parse_args()
    return args

def config(args):
    # GPU Config
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    elif args.gpu is not None and args.gpu > -1:
        torch.cuda.set_device(args.gpu)

    # Trim 
    if args.datadir[-1] == '/': args.datadir = args.datadir[:-1]

    # get kwargs
    kws = vars(args)
    if args.config and os.path.exists(kws['config']):
        with open(kws['config']) as f:
            config_kws = json.load(f)
            for k, v in config_kws.items():
                if v: kws[k] = v
            # kws.update(config_kws)
    return kws

# Initial setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def mixup_data(x, y, alpha=0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    mixed_x, y_a, y_b = map(Variable,(mixed_x, y_a, y_b))
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_loss(model, criterion, x,y,alpha):
    if alpha > 0:
        mix_x, ya, yb, lam = mixup_data(x,y,alpha)
        if torch.cuda.is_available():
            mix_x = mix_x.cuda()
            ya = ya.cuda()
            yb = yb.cuda()
        ya = ya.squeeze(1)
        yb = yb.squeeze(1)

        out = model(mix_x.unsqueeze(1))
        loss = mixup_criterion(criterion, out, ya, yb, lam)
        return loss
    else:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y = y.squeeze(1)
        out = model(x.unsqueeze(1))
        loss = criterion(out,y)
        return loss

from torch.optim.lr_scheduler import _LRScheduler
import warnings
class WarmStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, warm_step=20, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.warm_step = warm_step
        self.gamma = gamma
        super(WarmStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** ( 0 if self.last_epoch < self.warm_step else self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]




def train(kws, train_loader, val_dict, result_dir, target_names, train_y):
    print(f'Model: {kws["model"]}')
    shape = train_loader.dataset[0][0].shape  # Infer input shape from dataset
    model = getattr(models, kws['model'])(shape=shape, **kws)
    print(summary(model, torch.zeros((1, 1, *shape))))

    # Ensure model is on GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Set up criterion based on whether weights are provided
    if kws['weight']:
        count = Counter(train_y)
        nums = np.array([count['neutral'], count['happy'], count['sad'], count['angry']])
        weight = torch.Tensor(1 - nums / nums.sum())
        if torch.cuda.is_available():
            weight = weight.cuda()
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if kws['optimizer'] == 'adam':        
        optimizer = optim.RAdam(model.parameters(), lr=kws['learning_rate'], weight_decay=1e-6)
    elif kws['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=kws['learning_rate'], momentum=0.9, weight_decay=1e-4, nesterov=True)

    # Select learning rate scheduler
    if kws['lr_schedule'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    elif kws['lr_schedule'] == 'exp':
        scheduler = ExponentialLR(optimizer, 0.95)
    elif kws['lr_schedule'] == 'step':
        scheduler = StepLR(optimizer, 10, 0.1)
    elif kws['lr_schedule'] == 'warmstep':
        scheduler = WarmStepLR(optimizer, 10, 0.1)
    else:
        scheduler = StepLR(optimizer, 10000, 1)

    print("Training...")

    # Open log and loss files in result directory
    fh = open(os.path.join(result_dir, f'{kws["model"]}_train.log'), 'a', buffering=1)
    floss = open(os.path.join(result_dir, f'{kws["model"]}.loss'), 'a', buffering=1)
    
    maxACC = 0
    totalrunningTime = 0
    MODEL_PATH = os.path.join(result_dir, f'model_{kws["model"]}.pth')
    
    # Write initial seed to loss file
    floss.write(f'{kws["seed"]}\t')

    for i in range(kws['epochs']):
        startTime = time.perf_counter()
        tq = tqdm(total=len(train_loader))  # Progress bar
        model.train()  # Set model to training mode
        
        print_loss = 0
        for _, data in enumerate(train_loader):
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # Calculate loss
            loss = get_loss(model, criterion, x, y, kws['alpha'])

            print_loss += loss.data.item() * kws['batch_size']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.update(kws['batch_size'])
        tq.close()

        # Step scheduler if needed
        if optimizer.param_groups[0]['lr'] >= kws['lr_min']: 
            scheduler.step()

        floss.write(f'{print_loss / len(train_loader.dataset)}\t')
        floss.flush()
        
        print(f'Epoch: {i}, lr: {optimizer.param_groups[0]["lr"]:.4}, loss: {print_loss / len(train_loader.dataset):.4}')
        fh.write(f'Epoch: {i}, lr: {optimizer.param_groups[0]["lr"]:.4}, loss: {print_loss / len(train_loader.dataset):.4}\n')

        # Validation
        endTime = time.perf_counter()
        totalrunningTime += endTime - startTime
        fh.write(f'{totalrunningTime}\n')

        model.eval()  # Set model to evaluation mode
        y_true, y_pred = [], []
        for val in val_dict:
            x, y = val['X'], val['y']
            x = torch.from_numpy(x).float()
            y_true.append(y)
            if torch.cuda.is_available():
                x = x.cuda()
            if x.size(0) == 1:
                x = torch.cat((x, x), 0)
            out = model(x.unsqueeze(1))
            pred = out.mean(dim=0)
            pred = torch.max(pred, 0)[1].cpu().numpy()
            y_pred.append(int(pred))

        floss.write('\n')
        floss.flush()

        # Generate report and confusion matrix
        report = classification_report(y_true, y_pred, digits=6, target_names=target_names)
        report_dict = classification_report(y_true, y_pred, digits=6, target_names=target_names, output_dict=True)
        matrix = metrics.confusion_matrix(y_true, y_pred)

        WA = report_dict['accuracy'] * 100
        UA = report_dict['macro avg']['recall'] * 100
        macro_f1 = report_dict['macro avg']['f1-score'] * 100
        w_f1 = report_dict['weighted avg']['f1-score'] * 100

        ACC = (WA + UA) / 2
        if maxACC < ACC: 
            maxACC, WA_, UA_ = ACC, WA, UA
            macro_f1_, w_f1_ = macro_f1, w_f1
            best_re, best_ma = report, matrix
            if kws["SaveModel"]: 
                torch.save(model.state_dict(), MODEL_PATH)

        print(report)
        print(matrix)
        print(f'The best result ----------\nWA: {WA_:.4f}%, UA: {UA_:.4f}%, Macro F1: {macro_f1_:.4f}%, Weighted F1: {w_f1_:.4f}%\n--------------------------')

        # Write to log file
        fh.write(report)
        fh.write(f'{matrix}\n')
        fh.write(f'The best result ----------\n')
        fh.write(f'WA: {WA_:.4f}%, UA: {UA_:.4f}%, Macro F1: {macro_f1_:.4f}%, Weighted F1: {w_f1_:.4f}%\n')
        fh.write('--------------------------\n')

    floss.close()
    fh.close()

    return model, maxACC, WA_, UA_, macro_f1_, w_f1_




def test_model(model, test_dict, target_names):
    model.eval()  # Set model to evaluation mode
    y_true, y_pred = [], []
    
    with torch.no_grad():  # Disable gradient calculation during testing
        for test in test_dict:
            x, y = test['X'], test['y']
            
            # Transfer data to GPU if available
            x = torch.from_numpy(x).float()
            if torch.cuda.is_available():
                x = x.cuda()
                
            # Model forward pass
            out = model(x.unsqueeze(1))  # Ensure the input shape matches training
            
            # Take the average over time if required (sequence or batch dimension)
            pred = out.mean(dim=0).argmax().item()  # Get the predicted class (argmax)
            
            # Store predictions and true labels
            y_true.append(y)
            y_pred.append(int(pred))

    # Generate classification report
    report_dict = classification_report(y_true, y_pred, target_names=target_names, digits=6, output_dict=True)
    
    # Extract metrics
    WA = report_dict['accuracy'] * 100
    UA = report_dict['macro avg']['recall'] * 100
    macro_f1 = report_dict['macro avg']['f1-score'] * 100
    weighted_f1 = report_dict['weighted avg']['f1-score'] * 100

    # Print test results
    print(f"Test Results:\nWA: {WA:.4f}%, UA: {UA:.4f}%, Macro F1: {macro_f1:.4f}%, Weighted F1: {weighted_f1:.4f}%")
    
    # Return the important metrics and report
    return WA, UA, macro_f1, weighted_f1, report_dict





if __name__ == '__main__':
    args = parse_args()
    kws = config(args)

    # Load seeds from file
    with open(kws['seeds_file'], 'r') as f:
        seeds = [int(line.strip()) for line in f.readlines()]

    if not os.path.exists(kws["resultdir"]):
        os.makedirs(kws["resultdir"], exist_ok=True)

    # Define dataset conditions
    conditions = [
        ("clean", [""]),
        ("babble", ["-5", "0", "5", "10", "15", "20"]),
        ("noise", ["-5", "0", "5", "10", "15", "20"]),
        ("music", ["-5", "0", "5", "10", "15", "20"]),
        ("white", ["-5", "0", "5", "10", "15", "20"]),
        ("reverberation", ["largeroom"])
    ]
    for condition_name, sub_conditions in conditions:
        for sub_condition in sub_conditions:
            print(f"Processing {condition_name} - {sub_condition} dataset...")
            dataset_dir = os.path.join(kws['datadir'], condition_name, sub_condition)
            result_dir = os.path.join(kws['resultdir'], 'impro', condition_name, sub_condition)

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # Load features
            features_file = os.path.join(dataset_dir, f'features_mfcc_impro.pkl')
            with open(features_file, 'rb') as f:
                features = pickle.load(f)

            train_X, train_y, val_dict, test_dict = features['train_X'], features['train_y'], features['val_dict'], features['test_dict']
            train_data = data_loader.DataSet(train_X, train_y)



            
            # Collect results for each seed
            results = {'WA': [], 'UA': [], 'Macro F1': [], 'Weighted F1': []}
            log_file = os.path.join(result_dir, 'results_test.txt')

            with open(log_file, 'a') as f_log:
                for seed in seeds:
                    kws['seed'] = seed
                    setup_seed(seed)
                    train_loader = DataLoader(train_data, batch_size=kws['batch_size'], shuffle=True)

                    try:
                        # Train the model
                        model, maxACC, WA_, UA_, macro_f1_, w_f1_ = train(kws, train_loader, val_dict, result_dir, target_names, train_y)
                        
                        # Load the best saved model weights before testing
                        model_path = os.path.join(result_dir, f'model_{kws["model"]}.pth')
                        if os.path.exists(model_path):
                            model.load_state_dict(torch.load(model_path))
                            print(f"Loaded best model weights from {model_path}")
                        else:
                            print(f"Model weights not found at {model_path}. Skipping seed {seed}.")

                        # Test the model
                        WA, UA, macro_f1, weighted_f1, report_dict = test_model(model, test_dict, target_names)

                        # Save results for this seed
                        f_log.write(f'Test results for {condition_name} - {sub_condition} with seed {seed}:\n')
                        f_log.write(f'WA: {WA:.4f}%, UA: {UA:.4f}%, Macro F1: {macro_f1:.4f}%, Weighted F1: {weighted_f1:.4f}%\n')
                        f_log.write('--------------------------\n')
                        f_log.flush()

                        # Append results
                        results['WA'].append(WA)
                        results['UA'].append(UA)
                        results['Macro F1'].append(macro_f1)
                        results['Weighted F1'].append(weighted_f1)

                    except Exception as e:
                        print(f"Error during training or testing for seed {seed}: {e}")
                        f_log.write(f"Error during training/testing with seed {seed}: {str(e)}\n")
                        f_log.flush()

                    # Clear the model from memory after testing
                    del model
                    torch.cuda.empty_cache()  # Free up GPU memory

            # Compute average and std across all seeds
            avg_WA, std_WA = np.mean(results['WA']), np.std(results['WA'])
            avg_UA, std_UA = np.mean(results['UA']), np.std(results['UA'])
            avg_macro_f1, std_macro_f1 = np.mean(results['Macro F1']), np.std(results['Macro F1'])
            avg_weighted_f1, std_weighted_f1 = np.mean(results['Weighted F1']), np.std(results['Weighted F1'])

            # Save average and std results
            with open(log_file, 'a') as f_log:
                f_log.write(f'Average results for {condition_name} - {sub_condition} across all seeds:\n')
                f_log.write(f'WA: {avg_WA:.4f}% (±{std_WA:.4f}), UA: {avg_UA:.4f}% (±{std_UA:.4f}), Macro F1: {avg_macro_f1:.4f}% (±{std_macro_f1:.4f}), Weighted F1: {avg_weighted_f1:.4f}% (±{std_weighted_f1:.4f})\n')
                f_log.write('==========================\n')
                f_log.flush()  # Ensure results are saved immediately