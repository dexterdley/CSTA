import h5py
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import pdb
from decord import VideoReader, cpu
import math

def load_video_from_picks(video_path, picks):
    """
    Directly load the picked frames
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    picks_list = [min(pick, len(vr)-1) for pick in picks]
    frames = vr.get_batch(picks_list).asnumpy()
    return frames

def get_temporal_positional_encodings(num_frames, embedding_dim=128):
    """
    Generates the standard Sine/Cosine Positional Encoding matrix.
    """
    position = torch.arange(num_frames).unsqueeze(1).float() # (T, 1)
    
    # We only need half the dimensions for the frequencies
    i = torch.arange(0, embedding_dim, 2).float() # (D/2)
    
    # Denominator factor calculation
    exponent = i * (-math.log(10000.0) / embedding_dim)
    div_term = torch.exp(exponent) # (D/2)

    # Calculate the argument of the sine/cosine functions: (T, 1) * (1, D/2) -> (T, D/2)
    argument = position * div_term

    # Initialize the final encoding matrix
    encoding = torch.zeros(num_frames, embedding_dim)
    encoding[:, 0::2] = torch.sin(argument) # Apply Sine to even indices
    encoding[:, 1::2] = torch.cos(argument) # Apply Cosine to odd indices
    return encoding

class SumMeLLaMA_VideoDataset(Dataset):

    def __init__(self, 
                mode, 
                split_idx, 
                image_processor, 
                clip_length=16, 
                frame_stride=1,
                hidden_size=128,
                load_test=False,
                random_sampling=True):
        """
        SumMe Video Dataset
        Args:
            mode: 'train', 'val', or 'test'
            splits: sum splits
            image_processor: Processesor object for VIT
            clip_length: Number of frames per clip
            frame_stride: Stride between consecutive frames
            mode: 'train', 'val', or 'test'
            random_sampling: Whether to sample clips randomly
        """

        self.mode = mode
        self.clip_length = clip_length
        self.dataset = './SumMe/eccv16_dataset_summe_google_pool5.h5'
        self.split_file = './dataset/summe_splits.json'
        self.video_folder = './SumMe/raw/videos/'
        self.video_data = h5py.File(self.dataset, 'r')

        self.image_processor = image_processor
        self.frame_stride = frame_stride
        self.hidden_size = hidden_size
        self.load_test = load_test
        self.random_sampling = random_sampling

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())
            self.data = self.data[split_idx]
        
    def __len__(self):
        """ Function for the 'len' operator of dataset """
        return len(self.data[self.mode + '_keys'])

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

        video_filename = str(np.array(self.video_data[video_name + '/video_name']))
        video_filename = video_filename.strip("b'").strip('"').strip()

        if len(video_filename.split(" ")) > len(video_filename.split("_")):
            split_filename = video_filename.split(" ")
        else:
            split_filename = video_filename.split(" ")

        clean_filename = "".join([item + "_" for item in split_filename])
        video_path = self.video_folder + clean_filename

        if os.path.exists(video_path + ".webm"):
            video_path += ".webm"
        else:
            video_path += ".mp4"

        video_frames = load_video_from_picks(video_path, picks)
        video_sizes = [video_frames.shape[1], video_frames.shape[2]]
        total_frames = len(video_frames)
        title = video_filename

        if not self.load_test:
            # Training and Sample frame indices
            frame_indices = self._sample_frame_indices(total_frames)
            frames = self.image_processor.preprocess(video_frames[frame_indices], return_tensors="pt")["pixel_values"]
            gtscore = full_gtscore[frame_indices].unsqueeze(1)
            features = full_features[frame_indices].unsqueeze(1)
            timestamp_encodings = get_temporal_positional_encodings(total_frames, self.hidden_size)[frame_indices]

        else:
            gtscore = full_gtscore.unsqueeze(1)
            features = full_features.unsqueeze(1)
            frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
            timestamp_encodings = get_temporal_positional_encodings(total_frames, self.hidden_size)

        sample = {
            'video_name': video_name,
            'features': features,
            'gtscore': gtscore,
            'video': frames,
            'video_sizes': video_sizes,
            'title': title,
            'timestamp_encodings': timestamp_encodings
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
        video_name, video_filename, features, gtscore, video, video_sizes, title = [],[],[],[],[],[],[]
        timestamp_encodings = []
        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                video.append(data['video'])
                video_sizes.append(data['video_sizes'])
                title.append(data['title'])
                timestamp_encodings.append(data['timestamp_encodings'])

        except:
            print('Error in batch collator')

        frame_feat = pad_sequence(features, batch_first=True)
        video = pad_sequence(video, batch_first=True)
        # gtscore = pad_sequence(gtscore, batch_first=True)

        batch_data = {'video_name':video_name, 'features':frame_feat, 'gtscore':gtscore,
                      'video':video, 'video_sizes':video_sizes, 'title': title, 
                      'timestamp_encodings': timestamp_encodings}
        return batch_data

class ValBatchCollator(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, batch):
        video_name, video_filename, features, gtscore, video, video_sizes, title = [],[],[],[],[],[],[]
        timestamp_encodings = []
        cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                video.append(data['video'])
                video_sizes.append(data['video_sizes'])
                title.append(data['title'])
                timestamp_encodings.append(data['timestamp_encodings'])

                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator')

        frame_feat = pad_sequence(features, batch_first=True)
        video = pad_sequence(video, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)

        batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore,
                      'n_frames': n_frames, 'n_frame_per_seg': nseg, 'picks': picks, 'change_points': cps,
                      'video':video, 'video_sizes':video_sizes, 'title': title, 
                      'timestamp_encodings': timestamp_encodings, 'gt_summary': gt_summary}

        return batch_data