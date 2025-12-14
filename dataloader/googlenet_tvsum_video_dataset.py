import h5py
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import pdb

class TVSumGoogleNetDataset(Dataset):

    def __init__(self, 
                mode, 
                split_idx,
                clip_length=16, 
                frame_stride=1,
                load_test=False,
                random_sampling=True):
        """
        A simple baseline dataloader for TVSum Video Dataset
        returns only googlenet 'features', 'gtscore' and 'video_name'
        Args:
            mode: 'train', 'val', or 'test'
            splits: tvsum splits
        """

        self.mode = mode
        self.clip_length = clip_length
        self.dataset = './TVSum/eccv16_dataset_tvsum_google_pool5.h5'
        self.split_file = './dataset/tvsum_splits.json'
        self.video_folder = './tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/'
        self.video_data = h5py.File(self.dataset, 'r')

        self.info_file = './tvsum50_ver_1_1/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv'
        self.info_file = pd.read_csv(self.info_file, sep='\t')
        self.frame_stride = frame_stride
        self.load_test = load_test
        self.random_sampling = random_sampling

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())
            self.data = self.data[split_idx]

    def __len__(self):
        """ Function for te 'len' operator of dataset """
        return len(self.data[self.mode+'_keys'])

    def _sample_frame_indices(self, total_frames):
        """Sample frame indices for a clip"""
        if self.random_sampling:
            # Random sampling for training
            start_idx = np.random.randint(0, max(0, total_frames - self.clip_length * self.frame_stride))
        else:
            start_idx = max(0,(total_frames - self.clip_length * self.frame_stride) // 2) # Center sampling for validation/test

        # Generate all indices at once
        frame_indices = start_idx + np.arange(self.clip_length) * self.frame_stride
        
        # Clip indices that exceed total_frames
        frame_indices = np.minimum(frame_indices, total_frames - 1)

        return frame_indices.tolist()

    def __getitem__(self, index):
        video_name = self.data[self.mode + '_keys'][index]
        full_features = torch.as_tensor(self.video_data[video_name + '/features'])
        full_gtscore = torch.as_tensor(self.video_data[video_name + '/gtscore'])
        picks = self.video_data[video_name + '/picks']
        total_frames = len(picks)
        
        if not self.load_test:
            frame_indices = self._sample_frame_indices(total_frames)
            gtscore = full_gtscore[frame_indices].unsqueeze(1)
            features = full_features[frame_indices].unsqueeze(1)

        else:
            gtscore = full_gtscore.unsqueeze(1)
            features = full_features.unsqueeze(1)

        sample = {
            'video_name': video_name,
            'features': features,
            'gtscore': gtscore,
        }


        if self.mode != 'train':
            sample['picks'] = torch.as_tensor(np.array(picks))
            sample['n_frames'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frames']))
            sample['change_points'] = torch.as_tensor(np.array(self.video_data[video_name + '/change_points']))
            sample['n_frame_per_seg'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frame_per_seg']))
            sample['gt_summary'] = torch.as_tensor(np.array(self.video_data[video_name + '/user_summary']))

        return sample

class TrainBatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore = [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])

        except:
            print('Error in batch collator')

        features = pad_sequence(features, batch_first=True)

        batch_data = {'video_name':video_name, 'features':features, 'gtscore': gtscore}

        return batch_data

class ValBatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore = [], [], []
        cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])

                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator')

        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)

        batch_data = {'video_name':video_name, 'features' : frame_feat, 'gtscore':gtscore,
                      'n_frames': n_frames, 'n_frame_per_seg': nseg, 'picks': picks, 'change_points': cps, 'gt_summary': gt_summary}

        return batch_data