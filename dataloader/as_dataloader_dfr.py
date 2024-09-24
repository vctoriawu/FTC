#import os
from os.path import join
from random import randint, uniform
from typing import List, Dict, Union#, Optional, Callable, Iterable
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
#from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
import warnings
from random import lognormvariate
from random import seed
import torch.nn as nn
import random
from dataloader.utils import load_as_data, preprocess_as_data, fix_leakage
import imageio.v2 as imageio
import pdb
import cv2
from einops import rearrange, repeat

img_path_dataset = '/workspace/data/as_tom/annotations-all.csv'
tab_path_dataset = '/workspace/data/as_tom/finetuned_df.csv'
dataset_root = r"/workspace/data/as_tom"
cine_loader = 'mat_loader'

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

# for now, this only gets the matlab array loader
# Dict is a lookup table and can be expanded to add mpeg loader, etc
# returns a function
def get_loader(loader_name):
    loader_lookup_table = {'mat_loader': mat_loader}
    return loader_lookup_table[loader_name]

def mat_loader(path):
    mat = loadmat(path)
    if 'cine' in mat.keys():    
        return loadmat(path)['cine']
    if 'cropped' in mat.keys():    
        return loadmat(path)['cropped']

label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'normal': 0.0, 'mild': 1.0, 'moderate': 1.0, 'severe': 1.0},
    'all': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
    'not_severe': {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 1},
    'as_only': {'mild': 0, 'moderate': 1, 'severe': 2},
    'mild_moderate': {'mild': 0, 'moderate': 1},
    'moderate_severe': {'moderate': 0, 'severe': 1},
    'tufts': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 2}
}
class_labels: Dict[str, List[str]] = {
    'binary': ['Normal', 'AS'],
    'all': ['Normal', 'Mild', 'Moderate', 'Severe'],
    'not_severe': ['Not Severe', 'Severe'],
    'as_only': ['mild', 'moderate', 'severe'],
    'mild_moderate': ['mild', 'moderate'],
    'moderate_severe': ['moderate', 'severe'],
    'tufts': ['Normal', 'Early', 'Significant']
}

def save_videos_as_avi(video_tensor, video_path, frame_rate=30):
    video = rearrange(video_tensor, 'c t h w -> t h w c')
    if video.shape[-1] == 1:
        video = repeat(video, 't h w 1 -> t h w c', c=3)
    # Get the batch size, time_steps, height, width, channels from the tensor
    time_steps, height, width, _ = video.shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), frame_rate, (width, height))
    for t in range(time_steps):
        frame = (video[t]*255).cpu().numpy().astype(np.uint8) if isinstance(video, torch.Tensor) \
            else (video[t]*255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()    
    
def compute_intervals(df, unit, quantity):
    """
    Calculates the number of sub-videos from each video in the dataset
    Saves the frame window for each sub-video in a separate sheet

    Parameters
    ----------
    df : pd.DataFrame
        dataframe object containing frame rate, heart rate, etc.
    unit : str
        unit for interval retrieval, image/second/cycle
    quantity :
        quantity for interval retrieval,
        eg. 1.3 with "cycle" means each interval should be 1.3 cycles

    Returns
    -------
    df : pd.DataFrame
        updated dataframe containing num_intervals and window_size
    df_intervals: pd.DataFrame
        dataframe containing mapping between videos and window start/end frames

    """
    ms = df["frame_time"]
    hr = df["heart_rate"]
    if unit == "image":
        if int(quantity) < 1:
            raise ValueError("Must draw >= 1 image per video")
        df["window_size"] = int(quantity)
    elif unit == "second":
        df["window_size"] = (quantity * 1000 / ms).astype("int32")
    elif unit == "cycle":
        df["window_size"] = (quantity * 60000 / ms / hr).astype("int32")
    else:
        raise ValueError(f"Unit should be image/second/cycle, got {unit}")
    # if there are any window sizes of zero or less, raise an exception
    if len(df[df["window_size"] < 1]) > 0:
        raise Exception("Dataloader: Detected proposed window size of 0, exiting")

    df["num_intervals"] = (df["frames"] / df["window_size"]).astype("int32")

    video_idx, interval_idx, start_frame, end_frame = [], [], [], []
    for i in range(len(df)):
        video_info = df.iloc[i]
        if video_info["num_intervals"] == 0:
            video_idx.append(i)
            interval_idx.append(0)
            start_frame.append(0)
            end_frame.append(video_info["frames"])
        else:
            n_intervals = video_info["num_intervals"]
            w_size = video_info["window_size"]
            for j in range(n_intervals):
                video_idx.append(i)
                interval_idx.append(j)
                start_frame.append(j * w_size)
                end_frame.append((j + 1) * w_size)
    d = {
        "video_idx": video_idx,
        "interval_idx": interval_idx,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }
    df_interval = pd.DataFrame.from_dict(d)

    return df, df_interval

def get_dfr_dataloader(config, split, mode, dfr=False):
    '''
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : Configuration dictionary
        follows the format of get_config.py
    split : string, 'train'/'val'/'test' for which section to obtain
    mode : string, 'train'/'val'/'test' for setting augmentation/metadata ops

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits

    '''
    if mode=='train_val':
        flip=config['flip_rate']
        tra = True
        bsize = config['batch_size']
        show_info = False
    elif mode=='test':
        flip = 0.0
        tra = False
        bsize = 1
        show_info = True
    if show_info:
        assert bsize==1, "To show per-data info batch size must be 1"
    fr = 32
    
    # read in the data directory CSV as a pandas dataframe
    raw_dataset = pd.read_csv(img_path_dataset)
    dataset = pd.read_csv(img_path_dataset)
        
    # append dataset root to each path in the dataframe
    dataset['path'] = dataset['path'].map(lambda x: join(dataset_root, x))
    view = config['view']
        
    if view in ('plax', 'psax'):
        dataset = dataset[dataset['view'] == view]
    elif view != 'all':
        raise ValueError(f'View should be plax, psax or all, got {view}')
       
    # remove unnecessary columns in 'as_label' based on label scheme
    label_scheme_name = config['label_scheme_name']
    scheme = label_schemes[label_scheme_name]
    dataset = dataset[dataset['as_label'].isin( scheme.keys() )]

    ##load tabular dataset
    tab_train, tab_val, tab_test = load_as_data(csv_path = tab_path_dataset,
                                                drop_cols = config['drop_cols'],
                                                num_ex = config['num_ex'],
                                                scale_feats = config['scale_feats'])

    ##perform imputation 
    tab_trainset, tab_valset, tab_testset, all_cols = preprocess_as_data(tab_train, tab_val, tab_test, config['categorical_cols'])
    
    # Take train/test/val
    if mode=="train_val":
        train_dataset = dataset[dataset['split'] == "train"]
        val_dataset = dataset[dataset['split'] == "val"]
        #Fix data leakage 
        img_trainset = fix_leakage(df=raw_dataset, df_subset=train_dataset, split="train")
        img_valset = fix_leakage(df=raw_dataset, df_subset=val_dataset, split="val")
        #add new columns for dfr
        img_trainset = add_dfr_cols(img_trainset, scheme=scheme, n_places=2) 
        img_valset = add_dfr_cols(img_valset, scheme=scheme, n_places=2) 

        ## add train_set where bicuspid samples are from group 1 to val_set and get balanced val_set TODO
        dataset = balanced_trainval_set(img_trainset, img_valset)

        #combine tab_trainset and tab_valset
        tab_dataset = pd.concat([tab_trainset, tab_valset])
        
    elif mode=="test":
        test_dataset = dataset[dataset['split'] == "test"]
        tab_dataset = tab_testset
        #Fix data leakage 
        test_set = fix_leakage(df=raw_dataset, df_subset=test_dataset, split="test")
        #add new columns for dfr
        dataset = add_dfr_cols(dataset=test_set, scheme=scheme, n_places=2) #TODO

    else:
        raise ValueError(f'This mode is not supported, got {split}')
        
    dset = AorticStenosisDataset(img_path_dataset=dataset, 
                                tab_dataset=tab_dataset,
                                save_videos=config['save_videos'],
                                split=split,
                                transform=tra,
                                normalize=True,
                                frames=fr,
                                return_info=show_info,
                                contrastive_method = config['cotrastive_method'],
                                flip_rate=flip,
                                label_scheme = scheme,
                                dfr=dfr)
    
    loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=10)
    return loader

def add_dfr_cols(dataset, scheme, n_places=2): 
    y_ls = [scheme[x] for x in dataset['as_label']]
    y_arr = np.array(y_ls) 
    #add col for confounder_array/"p" (binary for bicuspid)
    dataset["confounder_arr"] = dataset['Bicuspid'].values 
    #add col for group_array
    dataset["group_array"] = (y_arr * n_places + dataset["confounder_arr"]).astype('int')

    return dataset

def balanced_trainval_set(train_set, val_set): 
    """add train_set where bicuspid samples are from group 1 to val_set and get balanced val_set""" 
    # get subset of train_df where group == 1
    bicuspid_trainset = train_set[train_set['group_array']==1].to_numpy()
    # get n_train_g1 --> length of train_df 
    n_train_g1 = len(bicuspid_trainset)
    # get min_g --> min number among groups in val_set
    min_g = val_set['group_array'].value_counts().min()
    n_groups = val_set['group_array'].nunique()

    # get pandas subsets for each group, then randomly select some rows that belong to each group until you have min_g
        #condition 1: for group 1, add the train_df consisting only of group 1 cases
    balanced_df = []
    for i, g in enumerate(range(n_groups)):
        if i==1:
            val_g1_idx = val_set[val_set['group_array']==1].index.to_numpy()
            np.random.shuffle(val_g1_idx)
            val_g1_df = val_set.loc[val_g1_idx[:min_g]].to_numpy()
            g1_df = np.concatenate((val_g1_df, bicuspid_trainset))
            balanced_df.append(g1_df)
        #condition 2: for other groups, add len(train_df) more rows randomly from each group
        else:
            g_idx = val_set[val_set['group_array']==i].index.to_numpy()
            np.random.shuffle(g_idx)
            g_idx_df = val_set.loc[g_idx[:min_g + n_train_g1]].to_numpy()
            balanced_df.append(g_idx_df)
    balanced_df = np.vstack(balanced_df)
    # combine subsets (rows will be shuffled later)
    balanced_df = pd.DataFrame(balanced_df, columns=val_set.columns) 

    return balanced_df

class AorticStenosisDataset(Dataset):
    def __init__(self, 
                 label_scheme,
                 img_path_dataset,
                 tab_dataset,
                 save_videos,
                 dfr,
                 split: str = 'train',
                 transform: bool = True, normalize: bool = True, 
                 frames: int = 16, resolution: int = 224,
                 cine_loader: str = 'mat_loader', return_info: bool = False, 
                 contrastive_method: str = 'CE',
                 flip_rate: float = 0.3, min_crop_ratio: float = 0.8, 
                 hr_mean: float = 4.237, hr_std: float = 0.1885,
                 **kwargs):

        self.return_info = return_info
        self.hr_mean = hr_mean
        self.hr_srd = hr_std
        self.scheme = label_scheme
        self.cine_loader = get_loader(cine_loader)
        self.get_nvideos = 0
        self.save_videos = save_videos
        self.dfr = dfr
        self.n_groups = img_path_dataset["group_array"].nunique()
        
        #From ProtoASNet
        self.interval_unit = "cycle"
        self.interval_quant = 1.0
        dataset, _ = compute_intervals(img_path_dataset, self.interval_unit, self.interval_quant)
        self.dataset = dataset

        self.tab_dataset = tab_dataset
        self.frames = frames
        self.resolution = (resolution, resolution)
        self.split = split
        self.transform = None
        self.transform_contrastive = None
        if transform:
            self.transform = Compose(
                [RandomResizedCrop(size=self.resolution, scale=(min_crop_ratio, 1)),
                 RandomHorizontalFlip(p=flip_rate)]
            )            
        self.normalize = normalize
        self.contrstive = contrastive_method    

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def get_random_interval(vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
    
    #@staticmethod
    def get_random_interval_protoasnet(self, vid_length, length):
        if length > vid_length:
            return 0, vid_length
        elif self.split == "test":
            return 0, length
        else:
            start = randint(0, vid_length - length)
            return start, start + length
    
    # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1, -1)
    
    # normalizes pixels based on pre-computed mean/std values
    @staticmethod
    def bin_to_norm(in_tensor):
        """
        normalizes the input tensor
        :param in_tensor: needs to be already in range of [0,1]
        """
        # in_tensor is 1xTxHxW
        m = 0.099
        std = 0.171
        return (in_tensor-m)/std

    def __getitem__(self, item):
        #pdb.set_trace()
        data_info = self.dataset.iloc[item]

        #get associated tabular data based on echo ID
        study_num = data_info['Echo ID#']
        tab_info = self.tab_dataset.loc[int(study_num)]
        tab_info = torch.tensor(tab_info.values, dtype=torch.float32)

        if self.dfr:
            group_arr = data_info["group_array"]
            places_arr = data_info["confounder_arr"]

        cine_original = self.cine_loader(data_info['path'])

        #if raw_dataset comes from all_cines in nas drive...
        folder = 'round2'
        if folder == 'all_cines':
            cine_original = cine_original.transpose((2,0,1))
        elif folder == 'round2':
            pass
            
        # window_length = 60000 / (lognormvariate(self.hr_mean, self.hr_srd) * data_info['frame_time'])
        # cine = self.get_random_interval(cine_original, window_length)
        #print(f"og cine shape: {cine_original.shape}\nrandom cine clip shape: {cine.shape}\n")
        ttd = 0.2
        window_length = max(int(data_info["window_size"] * uniform(1 - ttd, 1 + ttd)), 1)
        start_frame, end_frame = self.get_random_interval_protoasnet(data_info["frames"], window_length)
        cine = cine_original[start_frame:end_frame]
        cine = resize(cine, (self.frames, *self.resolution)) 
        cine = torch.tensor(cine).unsqueeze(0) #[1, 32, H, W]

        # save a sample of a video 
        if self.save_videos:
            if self.get_nvideos < 1:
                study_num = int(study_num)
                output_path = f'/workspace/miccai2024/videos/{study_num}.avi' # Output file name and format
                save_videos_as_avi(video_tensor=cine, video_path=output_path)
                self.get_nvideos +=1
                print(f"Saving video...")

        # storing labels as a dictionary will be in a future update
        if folder == 'round2':
            labels_B = torch.tensor(int(data_info['Bicuspid']))
        if folder == 'all_cines':
            labels_B = torch.tensor(int(2))
        labels_AS = torch.tensor(self.scheme[data_info['as_label']])

        if self.transform:
            if self.contrstive == 'CE' or self.contrstive == 'Linear':
                cine = self.transform(cine)
            else:
                cine_org = self.transform(cine)
                cine_aug = self.transform_contrastive(cine)
                cine = cine_org
                if random.random() < 0.4:
                    upsample = nn.Upsample(size=(16,224, 224), mode='nearest')
                    cine_aug = cine_aug[:, :,  0:180, 40:180].unsqueeze(1)
                    cine_aug = upsample(cine_aug).squeeze(1)   
                
        if self.normalize:
            cine = self.bin_to_norm(cine)
            if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='all'):
                cine_aug = self.bin_to_norm(cine_aug)  

        cine = self.gray_to_gray3(cine) #[3, F, H, W]
        cine = cine.float()
        
        if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='train_all'):
            cine_aug = self.gray_to_gray3(cine_aug)
            cine_aug = cine_aug.float()
            
        
        # slowFast input transformation
        #cine = self.pack_transform(cine)
        if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='train_all'):
            ret = ([cine,cine_aug], tab_info, labels_AS, labels_B)
       
        else:
            if self.dfr:
                ret = (cine, tab_info, labels_AS, labels_B), (places_arr, group_arr) 
            else:
                ret = (cine, tab_info, labels_AS, labels_B)
        if self.return_info:
            di = data_info.to_dict()
            di['window_length'] = window_length
            di['original_length'] = cine_original.shape[1]
            if self.dfr:
                ret = (cine, tab_info, labels_AS, labels_B, di, cine_original), (places_arr, group_arr) 
            else:
                ret = (cine, tab_info, labels_AS, labels_B, di, cine_original)

        return ret