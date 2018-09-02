
# coding: utf-8

# Before running this code, make sure you have <br>
# 1) images/  folder which as all the images (raw data; unzipped from the tar file) <br>
# 2) images/train_val_filtered.pkl : list of non blacklisted training images <br>
# 3) images/test_filtered.pkl : list of non blacklisted test images

# In[1]:
#1epo

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import pickle

# In[2]:

from sklearn.metrics import confusion_matrix
from sklearn.metrics.ranking import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer


# In[3]:


import pandas as pd
import numpy as np
import time


# In[5]:


def label2vector(label):

    ## INPUT : ['Atelectasis', 'Cardiomegaly']
    ## OUTPUT : [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    label = [label]
    test_labels_str = [i.split('|') for i in label]
    test_labels_str = [ x if 'Finding' not in x[0] else [] for x in (test_labels_str)]
    return mlb.transform(test_labels_str)[0]

def get_vector_labels(imagepath= '../data/images/', is_test = False):

    ## INPUT : ('../data/images/, True)
    ## OUTPUT :

#         img_filename               vector
# 44187   00013774_026.png    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# 44188   00013774_028.png    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


    ## read list of non-blacklisted images; train and test
    if not is_test:
        train_img_list = pd.read_pickle("train_val_filtered.pkl")
    else:
        train_img_list = pd.read_pickle("test_filtered.pkl")

    ## read images in our images folder
    single_tar_images = os.listdir(imagepath)
    ## subset
    train_img_list = train_img_list[train_img_list['img_filename'].isin(single_tar_images)]

    train_img_list['vector'] = train_img_list['text_label'].apply(label2vector)
    train_img_list = train_img_list[['img_filename', 'vector']]
    return train_img_list

def train_val_splitter(train_df, percentage = 0.125):

    assert percentage <= 1
    assert percentage > 0

    ## INPUT :  training_df of the shape (n,2), 0 <float <=1
    ## OUTPUT : validation_df, train_df; no patient overlap in the two dfs
                # validation_Df.shape = ~(n*percentage,2),
                # train_df.shape = ~(n*(1-percentage), 2)
    np.random.seed(1)
    col_list = train_df.columns.tolist()

    train_df['patient'] = train_df.img_filename.apply(lambda x : x.split("_")[0])
    valid_patients = np.random.choice(train_df.patient.unique(),
                 int(train_df.patient.unique().shape[0]*percentage),
                 replace=False)

    valid_df = train_df[train_df['patient'].isin(valid_patients)]
    dummy = train_df[~train_df['patient'].isin(valid_patients)]

    # assert no rows are missed
    assert valid_df.shape[0] + dummy.shape[0] == train_df.shape[0]
    # assert intersection is null
    assert np.intersect1d(valid_df.patient.values, dummy.patient.values).tolist() == []

    return valid_df[col_list], dummy[col_list]


# In[14]:


class XRayDataset(Dataset):

    def __init__(self, train_df, imagepath, transform=None):
        self.df = train_df.reset_index(drop = True)
        self.imagepath = imagepath
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.imagepath,
                                self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        labels = torch.FloatTensor(self.df[self.df['img_filename'] == self.df.iloc[idx, 0]]['vector'].values[0])
        #labels = torch.FloatTensor(np.array([0]*14))

        if self.transform is not None:
            image = self.transform(image)

        return image, labels


# In[16]:


def get_symbol(out_features=14):
    model = models.densenet.densenet121(pretrained=True)
    # Replace classifier (FC-1000) with (FC-14)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, out_features),
        nn.Sigmoid())
    # CUDA
    model.cuda()
    return model


# In[17]:


def init_symbol(sym, lr=0.0001):
    # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    opt = optim.Adam(sym.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    scheduler = ReduceLROnPlateau(opt, factor = 0.1, patience = 5, mode = 'min')
    return opt, criterion, scheduler


# In[18]:


def compute_roc_auc(data_gt, data_pd, mean=True, classes=14):
    roc_auc = []
    data_gt = data_gt.cpu().numpy()
    data_pd = data_pd.cpu().numpy()
    for i in range(classes):
        roc_auc.append(roc_auc_score(data_gt[:, i], data_pd[:, i]))
    if mean:
        roc_auc = np.mean(roc_auc)
    return roc_auc


# In[19]:


def train_epoch(model, dataloader, optimizer, criterion, epoch, batch=32):
    model.train()
    print("Training epoch {}".format(epoch+1))
    loss_val = 0
    loss_cnt = 0
    for data, target in (dataloader):
        # Get samples
        data = Variable(torch.FloatTensor(data).cuda())
        target = Variable(torch.FloatTensor(target).cuda())
        # Init
        optimizer.zero_grad()
        # Forwards
        output = model(data)
        # Loss
        loss = criterion(output, target)
        # Back-prop
        loss.backward()
        optimizer.step()
         # Log the loss
        loss_val += loss.data[0]
        loss_cnt += 1
    print("Training loss: {0:.4f}".format(loss_val/loss_cnt))


def valid_epoch(model, dataloader, criterion, epoch, phase='valid', batch=32):
    model.eval()
    if phase == 'testing':
        print("Testing epoch {}".format(epoch+1))
    else:
        print("Validating epoch {}".format(epoch+1))
    out_pred = torch.FloatTensor().cuda()
    out_gt = torch.FloatTensor().cuda()
    loss_val = 0
    loss_cnt = 0
    for data, target in dataloader:
        # Get samples
        data = Variable(torch.FloatTensor(data).cuda(), volatile=True)
        target = Variable(torch.FloatTensor(target).cuda(), volatile=True)
         # Forwards
        output = model(data)
        # Loss
        loss = criterion(output, target)
        # Log the loss
        loss_val += loss.data[0]
        loss_cnt += 1
        # Log for AUC
        out_pred = torch.cat((out_pred, output.data), 0)
        out_gt = torch.cat((out_gt, target.data), 0)
    loss_mean = loss_val/loss_cnt
    if phase == 'testing':
        print("Test-Dataset loss: {0:.4f}".format(loss_mean))
        print("Test-Dataset AUC: {}".format(compute_roc_auc(out_gt, out_pred, mean = False)) )
        return (loss_mean, compute_roc_auc(out_gt, out_pred, mean = False))
    else:
        print("Validation loss: {0:.4f}".format(loss_mean))
        print("Validation AUC: {}".format(compute_roc_auc(out_gt, out_pred, mean = False)))
        return (loss_mean, compute_roc_auc(out_gt, out_pred, mean = False))

def print_learning_rate(opt):
    for param_group in opt.param_groups:
        print("Learining rate: ", param_group['lr'])


#%%

if __name__ == '__main__':

    assert torch.cuda.is_available(), 'No GPU is configured for torch'


    set_labels =['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                 'Fibrosis','Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                 'Pneumonia','Pneumothorax']

    ## convert to Multilabels
    mlb = MultiLabelBinarizer(classes=set_labels)
    mlb.fit(set_labels)

    train_df = get_vector_labels()


    test_df = get_vector_labels(is_test = True)

    valid_df, train_df = train_val_splitter(train_df, 0.125)
    print (valid_df.shape, train_df.shape, test_df.shape)
    CLASSES = 14
    WIDTH = 224
    HEIGHT = 224
    LR = 0.0001
    EPOCHS = 1
    # Can scale to max for inference but for training LR will be affected
    # Prob better to increase this though on P100 since LR is not too low
    # Easier to see when plotted
    BATCHSIZE = 64 #64*2
    IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_RGB_SD = [0.229, 0.224, 0.225]
    imagepath = '../data/images/'

#%%
    normalize = transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD)

    train_dataset = XRayDataset(train_df,
                                imagepath=imagepath,
                                transform=transforms.Compose([
                                transforms.Resize(264),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop(size=WIDTH),
                                transforms.ColorJitter(0.15, 0.15),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),  # need to convert image to tensor!
                                normalize]))

    test_dataset = XRayDataset(test_df,
                               imagepath=imagepath,
                               transform=transforms.Compose([
                               transforms.Resize(WIDTH),
                               transforms.ToTensor(),
                               normalize]))

    valid_dataset = XRayDataset(valid_df,
                                imagepath=imagepath,
                                transform=transforms.Compose([
                                transforms.Resize(WIDTH),
                                transforms.ToTensor(),
                                normalize]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE,
                              shuffle=True, num_workers=24, pin_memory=False)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=8*BATCHSIZE,
                              shuffle=False, num_workers=24, pin_memory=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=8*BATCHSIZE,
                             shuffle=False, num_workers=24, pin_memory=False)
    #load pickled model instead of downloading, no internet in Gatech!
    with open('densenet_model.pkl', 'rb') as f:
        mms_model = pickle.load(f)
    print('model loaded')
    mms_model.cuda()

    optimizer, criterion, scheduler = init_symbol(mms_model)

    loss_min = float("inf")
    valid_loss = []

    # No-training
    valid_epoch(mms_model, valid_loader, criterion, -1)

    # Main train/val/test loop
    for j in (range(EPOCHS)):
        stime = time.time()
        print('training epoch:',j)
        train_epoch(mms_model, train_loader, optimizer, criterion, j)
        loss_val, loss_val2 = valid_epoch(mms_model, valid_loader, criterion, j)
        valid_loss.append(loss_val2)
        # LR Schedule
        scheduler.step(loss_val)
        print_learning_rate(optimizer)
        # todo: tensorboard hooks
        # Logging
        if j == max(range(EPOCHS)):
            print("last epoch. Saving ...")
            loss_min = loss_val
            torch.save(mms_model, 'Model_all_50_4-23-18.model')
            pd.DataFrame(valid_loss).to_csv("valid_loss_all_50_4-23-18.csv")
        etime = time.time()
        print("Epoch time: {0:.0f} seconds".format(etime-stime))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
