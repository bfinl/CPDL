'''
This is the main training paradigm for DL models used in the manuscript: "Continuous Tracking using Deep Learning-based Decoding for Non-invasive Brain-Computer Interface".

D. Forenzo, H. Zhu, J. Shanahan, J. Lim, and B. He, “Continuous Tracking using Deep Learning-based Decoding for Non-invasive Brain-Computer Interface.” bioRxiv, p. 2023.10.12.562084, Oct. 17, 2023. doi: 10.1101/2023.10.12.562084.

Takes in 2 arguments:
    1: DL model type, either EEGNet or PointNet
    2: Subject and session in the format: S##-Se## (ex. S01_Se02 for subject 1 session 2)

Model inputs are stored in .pkl files containing lists of 1 second long windows of EEG data in the format of [#channels x # of time points], with matching lists of data labels in the format [horizontal label, vertical label]

Author: Hao Zhu
'''
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
import numpy as np
from models import SimpleNet, EEGNetv4Like, PointNet2, PNtest, MyPNtest
from dataset.dataloader import getDatasets, CPDataset_NonSeq, CPDataset_NonSeq_Temporal, CPDataset4PointNet, CPDataset4PointNet_Temporal
from util import buf_centralize
from tqdm import tqdm
import sys, pickle, joblib, logging, os, re
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info('This is a log info')

def get_data_from_file(trial_xkfunc, num_files=100): #Function to grab relevent data from pkl files
    cnt = num_files
    Dataset, Task = [], []
    splitted_path = "pklFileLocation/"
    filelist = sorted(os.listdir(splitted_path))
    #Loop through available files and only take relevent data
    for file in filelist:
        subj, sess, mdl, trial = re.search(r"(\d+)-(\d+)-([A-Z]+)-(\d+).pkl", file).groups()
        subj = int(subj)
        trial = int(trial)
        sess = int(sess)
        if trial_xkfunc(subj, sess, mdl, trial): #Check if file relevent to current training
            with open(os.path.join(splitted_path, file), 'rb') as f:
                dataset, task = joblib.load(f)
            Dataset.append(dataset)
            Task.append(task)
            cnt -= 1
            if cnt == 0:
                yield Dataset, Task
                cnt = num_files
                Dataset, Task = [], []
    if cnt != num_files:
        yield Dataset, Task

def calDifAng(pred, label): #Function to calculate ADA from prediction and label vector lists
    pred = pred.squeeze()
    label = label.squeeze()
    assert len(pred.shape) == 2 and pred.shape[1] == 2
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()
    #Ignore cases where both prediction and label are zero length
    idx = ((pred**2).sum(axis=1) > 0) & ((label**2).sum(axis=1) > 0)
    pred = pred[idx]
    label = label[idx]
    #Calculate angle between prediction and label vectors
    coss = (pred*label).sum(axis=1)/(pred**2).sum(axis=1)**.5/(label**2).sum(axis=1)**.5
    coss[coss>1] = 1
    coss[coss<-1] = -1
    return (np.arccos(coss)/np.pi*180).mean().item()

def my_loss(pred, label):
    pred_cov = torch.cov(pred.T)
    pred_mu = pred.mean(dim=0)
    #MSE loss with additional KL divergence term
    return 1*F.mse_loss(pred, label) - torch.log(torch.det(pred_cov)) + torch.trace(pred_cov) + 2*pred_mu@pred_mu

#Check for GPU
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def trainOneSubj(Subj):
    #Main training function
    #Get subject and session information from argument
    subj, sess = re.search(r"S(\d+)-Se(\d+)", Subj).groups()
    subj = int(subj)
    sess = int(sess)
    #Each model was used for 20 trials in each session (4 runs)
    #Take first 15 trials as training data, last 5 as validation set
    train_func = lambda a, b, c, d: a == subj and b < sess and d%20 < 15
    valid_func = lambda a, b, c, d: a == subj and b < sess and d%20 >= 15
    max_epoch = 15

    #Select model type using argument
    mdl = sys.argv[1]
    if mdl == 'EEGNet':
        datatyp = 'temporal'
        CPDataset = CPDataset_NonSeq_Temporal
        model = EEGNetv4Like(62,2,input_window_samples=250)
        save_path = f'pathToSave/S{str(subj).zfill(2)}-EG-Se{str(sess).zfill(2)}.pth'
        lr = 0.005
        wd = 0
    elif mdl == 'PointNet':
        datatyp = 'frequency'
        CPDataset = CPDataset4PointNet
        model = PointNet2(2)
        save_path = f'pathToSave/S{str(subj).zfill(2)}-PN-Se{str(sess).zfill(2)}.pth'
        lr = 0.001
        wd = 1e-4
    model.to(device)

    #Setup training parameters
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch-1)
    best_valid = 900
    final_res = 90

    #Main training loop
    for _ in range(max_epoch):
        #Do training
        model.train()
        train_preds = []
        train_labels = []
        for trainDataset, trainTask in get_data_from_file(train_func, 100):
            train_dataset = CPDataset(trainDataset, trainTask, subj=Subj, stage='train')
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True,
                collate_fn=lambda x: [y.to(device) for y in default_collate(x)])
            for data, label in tqdm(train_loader):
                optimizer.zero_grad()
                pred = model(data)
                train_preds.append(pred.detach().cpu())
                train_labels.append(label.cpu())
                loss = my_loss(pred, label)
                loss.backward()
                optimizer.step()
        scheduler.step()

        #Do validation
        model.eval()
        valid_preds = []
        valid_labels = []
        for validDataset, validTask in get_data_from_file(valid_func, 100):
            valid_dataset = CPDataset(validDataset, validTask, subj=Subj, stage='valid')
            valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,
                collate_fn=lambda x: [y.to(device) for y in default_collate(x)])
            for data, label in valid_loader:
                pred = model(data)
                valid_preds.append(pred.detach().cpu())
                valid_labels.append(label.cpu())
        
        #Calculate metrics
        train_dif = calDifAng(np.concatenate(train_preds), np.concatenate(train_labels))
        valid_dif = calDifAng(np.concatenate(valid_preds), np.concatenate(valid_labels))
        print('{} {}:'.format(Subj, mdl), 'epoch ', _, '{:.2f}'.format(train_dif), '{:.2f}'.format(valid_dif))
        #Check if validation performance is best so far
        if best_valid > valid_dif:
            #If so, save model weights
            torch.save(model.state_dict(), save_path)
            best_valid = valid_dif

    return best_valid

#Main
for Subj in [sys.argv[2]]:
    trainOneSubj(Subj)