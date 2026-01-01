import h5py
import numpy as np
import torch
import torch.nn as nn
import warnings
import os
from tqdm import tqdm

from config import get_config
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from utils import print_args, get_gt
from torch.utils.data import DataLoader

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from dataloader.llava_summe_video_dataset import SumMeLLaMA_VideoDataset, ValBatchCollator as SumMeValCollator
from dataloader.llava_tvsum_video_dataset import TVSumLLaMA_VideoDataset, ValBatchCollator as TVSumValCollator

# Suppress warning regarding tensor creation from list of numpy arrays
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")

# Load configurations
config = get_config()
config.model_name = 'LLaVAVLS'
config.output_name = 'lstm'

custom_system_instruction = (
    "You are an intelligent chatbot designed to critically assess the importance score of each frame within a video context. "
    "Evaluate each frame using the following criteria: "
    "1. Narrative Significance "
    "2. Uniqueness and Novelty "
    "3. Action and Dynamics"
)

class LSTMAttnModel(nn.Module):
    def __init__(self, config_size):
        super().__init__()
        bottleneck_size = 256
        self.hidden_dim = bottleneck_size * 2  # Bi-LSTM output is hidden_size

        # 1. Temporal modeling with Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=config_size, #+ 256,
            hidden_size=bottleneck_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 2. Final Classification Head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        self._reset_parameters()
        
    def _reset_parameters(self):
        for module in self.mlp_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        return self.mlp_head(x)

def get_llava_output(config, llava_model, tokenizer, batch, device):
    question = getattr(config, 'prompt', "Please summarize the video.")
    if question is None: question = "Please summarize the video."
    
    # Ensure video is on the correct device and dtype
    video = batch['video'].half().to(device)
    video_sizes = batch['video_sizes'][0]

    video = video.view(-1, *video.shape[2:])
    vid_len = video.shape[0]

    qs = question

    if llava_model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = getattr(config, 'conv_mode', "vicuna_v1")
    conv = conv_templates[conv_mode].copy()
    conv.system = custom_system_instruction
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    if tokenizer.pad_token_id is None:
            if "qwen" in tokenizer.name_or_path.lower():
                tokenizer.pad_token_id = 151643

    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()

    all_hidden_states = []
    chunk_size = 32
    for i in range(0, vid_len, chunk_size):
        chunk_video = video[i:i+chunk_size]
        chunk_len = chunk_video.shape[0]
        chunk_input_ids = input_ids.repeat(chunk_len, 1)
        chunk_attention_masks = attention_masks.repeat(chunk_len, 1)

        with torch.no_grad():
            logits, _, hidden_states = llava_model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_masks,
                images=chunk_video,
                image_sizes=[[item] * getattr(config, 'clip_length', 8) for item in video_sizes],
                modalities=['image'] * chunk_len,
                dpo_forward=True
            )
        all_hidden_states.append(hidden_states.mean(1).float().cpu())
        del logits, hidden_states, chunk_video, chunk_input_ids, chunk_attention_masks
        torch.cuda.empty_cache()

    last_hidden_states = torch.cat(all_hidden_states, dim=0).to(device)
    return last_hidden_states

def main():
    print_args(config)
    device = torch.device(config.device)

    # Load LLaVA model
    print("--> Loading LLaVA model...")
    model_path = getattr(config, 'model_path', "lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    model_base = getattr(config, 'model_base', None)
    model_name = get_model_name_from_path(model_path)
    load_8bit = getattr(config, 'load_8bit', False)
    tokenizer, llava_model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name,
        device_map=None, load_8bit=load_8bit, attn_implementation="eager"
    )
    
    llava_model = llava_model.to(device)
    llava_model.eval()
    print("OK LLaVA model loaded.")

    for dataset in config.datasets:
        user_scores = get_gt(dataset)
        split_kendalls = []
        split_spears = []

        print(f"Processing {dataset}...")

        for split_id in range(5):
            # Setup DataLoader
            if dataset == 'SumMe':
                DatasetCls = SumMeLLaMA_VideoDataset
                ValCollator = SumMeValCollator
            elif dataset == 'TVSum':
                DatasetCls = TVSumLLaMA_VideoDataset
                ValCollator = TVSumValCollator
            
            clip_length = getattr(config, 'clip_length', 8)
            test_ds = DatasetCls(mode='test', split_idx=split_id, image_processor=image_processor, clip_length=clip_length, hidden_size=512, load_test=True)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=ValCollator(mode="val"), pin_memory=True)

            # Setup Attn Model
            if hasattr(config, 'output_name') and "lstm" in config.output_name:
                attn_model = LSTMAttnModel(llava_model.config.hidden_size)
            else:
                attn_model = AttnModel(llava_model.config.hidden_size)
            
            # Load weights
            weight_path = f'./weights/{dataset}/split{split_id+1}_lstm_attn_model.pth'
            if not os.path.exists(weight_path):
                print(f"Weights not found: {weight_path}. Skipping split {split_id}.")
                continue
            
            print(f"Loading weights from {weight_path}")
            attn_model.load_state_dict(torch.load(weight_path, map_location='cpu'))
            attn_model.to(device)
            attn_model.eval()

            kendalls = []
            spears = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Split {split_id}", leave=False):
                    # Get features from LLaVA
                    hidden_states = get_llava_output(config, llava_model, tokenizer, batch, device)
                    
                    # Get scores from AttnModel
                    output = attn_model(hidden_states)
                    
                    video_num = batch['video_name'][0]
                    
                    with h5py.File(f'./data/eccv16_dataset_{dataset.lower()}_google_pool5.h5','r') as hdf:
                        user_summary = np.array(hdf[video_num]['user_summary'])
                        sb = np.array(hdf[f"{video_num}/change_points"])
                        n_frames = np.array(hdf[f"{video_num}/n_frames"])
                        positions = np.array(hdf[f"{video_num}/picks"])
                    
                    scores = output.squeeze().clone().detach().cpu().numpy().tolist()
                    summary = generate_summary([sb], [scores], [n_frames], [positions])[0]

                    if dataset=='SumMe':
                        spear,kendall = get_corr_coeff([summary],[video_num],dataset,user_summary)
                    elif dataset=='TVSum':
                        spear,kendall = get_corr_coeff([scores],[video_num],dataset,user_scores)
                    
                    spears.append(spear)
                    kendalls.append(kendall)
            
            avg_kendall = np.mean(kendalls)
            avg_spear = np.mean(spears)
            split_kendalls.append(avg_kendall)
            split_spears.append(avg_spear)
            
            print("[Split{}] Kendall:{:.3f}, Spear:{:.3f}".format(split_id, avg_kendall, avg_spear))

        print("[FINAL - {}] Kendall:{:.3f}, Spear:{:.3f}".format(
            dataset, np.mean(split_kendalls), np.mean(split_spears)
        ))
        print()

if __name__ == "__main__":
    main()
