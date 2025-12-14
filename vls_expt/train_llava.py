import h5py
import numpy as np
import shutil
import torch
import torch.nn as nn

from tqdm import tqdm

from config import get_config
from dataset import create_dataloader
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from model import set_model
from utils import report_params, print_args, get_gt
from torch.utils.data import DataLoader

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from dataloader.llava_summe_video_dataset import SumMeLLaMA_VideoDataset, TrainBatchCollator as SumMeTrainCollator, ValBatchCollator as SumMeValCollator
from dataloader.llava_tvsum_video_dataset import TVSumLLaMA_VideoDataset, TrainBatchCollator as TVSumTrainCollator, ValBatchCollator as TVSumValCollator

# Load configurations
config = get_config()
config.model_name = 'LLaVAVLS'
# Print information of setting
print_args(config)

custom_system_instruction = (
    "You are an intelligent chatbot designed to critically assess the importance score of each frame within a video context. "
    "Evaluate each frame using the following criteria: "
    "1. Narrative Significance "
    "2. Uniqueness and Novelty "
    "3. Action and Dynamics"
)

class AttnModel(nn.Module):
    def __init__(self, config_size):
        super().__init__()
        bottleneck_size = 512
        
        self.mlp_head = nn.Sequential(
            nn.Linear(config_size, bottleneck_size),
            nn.LayerNorm(bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, bottleneck_size//2),
            nn.ReLU(),
            # 2. Final scoring layer
            nn.Linear(bottleneck_size//2, 1),
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
        return self.mlp_head(x)

class LSTMAttnModel(nn.Module):
    def __init__(self, config_size):
        super().__init__()
        bottleneck_size = 384
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
        x, _ = self.lstm(x)
        return self.mlp_head(x)

class LLaVAVLS(nn.Module):
    def __init__(self, config, llava_model, tokenizer):
        super().__init__()
        self.config = config
        self.llava_model = llava_model
        self.tokenizer = tokenizer

        # Freeze LLaVA model
        self.llava_model.eval()
        for param in self.llava_model.parameters():
            param.requires_grad = False

        # Trainable video summarization model
        if hasattr(self.config, 'output_name') and "lstm" in self.config.output_name:
            self.attn_model = LSTMAttnModel(self.llava_model.config.hidden_size).to(self.llava_model.device)
        else:
            self.attn_model = AttnModel(self.llava_model.config.hidden_size).to(self.llava_model.device)

    def forward(self, batch):
        question = getattr(self.config, 'prompt', "Please summarize the video.")
        if question is None: question = "Please summarize the video."
        video = batch['video'].half()
        video_sizes = batch['video_sizes'][0]

        video = video.view(-1, *video.shape[2:])
        vid_len = video.shape[0]

        qs = question

        if self.llava_model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = getattr(self.config, 'conv_mode', "vicuna_v1")
        conv = conv_templates[conv_mode].copy()
        conv.system = custom_system_instruction
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.llava_model.device)

        if self.tokenizer.pad_token_id is None:
                if "qwen" in self.tokenizer.name_or_path.lower():
                    # print("Setting pad token to bos token for qwen model.")
                    self.tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            logits, _, hidden_states = self.llava_model(
                input_ids=input_ids.repeat(vid_len, 1),
                attention_mask=attention_masks.repeat(vid_len, 1),
                images=video,
                image_sizes=[[item] * getattr(self.config, 'clip_length', 8) for item in video_sizes],
                modalities=['image'] * vid_len,
                dpo_forward=True
            )

        last_hidden_states = hidden_states[:, -1].float()
        
        impt_scores = self.attn_model(last_hidden_states)
        return impt_scores

# Start training
llava_model = None
tokenizer = None
image_processor = None

if config.model_name == 'LLaVAVLS':
    print("--> Loading LLaVA model...")
    model_path = getattr(config, 'model_path', "lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    model_base = getattr(config, 'model_base', None)
    model_name = get_model_name_from_path(model_path)
    load_8bit = getattr(config, 'load_8bit', False)
    tokenizer, llava_model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name,
        device_map=None, load_8bit=load_8bit, attn_implementation="eager"
    )
    print("OK LLaVA model loaded.")

for dataset in tqdm(config.datasets,total=len(config.datasets),ncols=70,leave=True,position=0):
    user_scores = get_gt(dataset)

    if dataset=='SumMe':
        batch_size = 1 if config.batch_size=='1' else int(config.SumMe_len*0.8*float(config.batch_size))
    elif dataset=='TVSum':
        batch_size = 1 if config.batch_size=='1' else int(config.TVSum_len*0.8*float(config.batch_size))

    if config.model_name == 'LLaVAVLS':
        def create_llava_splits():
            for i in range(5):
                if dataset == 'SumMe':
                    DatasetCls = SumMeLLaMA_VideoDataset
                    TrainCollator = SumMeTrainCollator
                    ValCollator = SumMeValCollator
                elif dataset == 'TVSum':
                    DatasetCls = TVSumLLaMA_VideoDataset
                    TrainCollator = TVSumTrainCollator
                    ValCollator = TVSumValCollator
                
                clip_length = getattr(config, 'clip_length', 8)
                
                train_ds = DatasetCls(mode='train', split_idx=i, image_processor=image_processor, clip_length=clip_length, hidden_size=512, load_test=False)
                test_ds = DatasetCls(mode='test', split_idx=i, image_processor=image_processor, clip_length=clip_length, hidden_size=512, load_test=False)
                
                bs = 1 # Force batch size 1 for LLaVA

                train_l = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, collate_fn=TrainCollator(), pin_memory=True)
                test_l = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=ValCollator(mode="val"), pin_memory=True)
                yield i, (train_l, test_l)
        dataloader_iterator = create_llava_splits()
    else:
        dataloader_iterator = enumerate(create_dataloader(dataset))

    for split_id,(train_loader,test_loader) in tqdm(dataloader_iterator,total=5,ncols=70,leave=False,position=1,desc=dataset):
        if config.model_name == 'LLaVAVLS':
            model = LLaVAVLS(config, llava_model, tokenizer)
        else:
            model = set_model(
            model_name=config.model_name,
            Scale=config.Scale,
            Softmax_axis=config.Softmax_axis,
            Balance=config.Balance,
            Positional_encoding=config.Positional_encoding,
            Positional_encoding_shape=config.Positional_encoding_shape,
            Positional_encoding_way=config.Positional_encoding_way,
            Dropout_on=config.Dropout_on,
            Dropout_ratio=config.Dropout_ratio,
            Classifier_on=config.Classifier_on,
            CLS_on=config.CLS_on,
            CLS_mix=config.CLS_mix,
            key_value_emb=config.key_value_emb,
            Skip_connection=config.Skip_connection,
            Layernorm=config.Layernorm
            )
        model.to(config.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=float(config.learning_rate),weight_decay=float(config.weight_decay))

        model_selection_kendall = -1    
        model_selection_spear = -1

        for epoch in tqdm(range(config.epochs),total=config.epochs,ncols=70,leave=False,position=2,desc=f'Split{split_id+1}'):
            model.train()
            update_loss = 0.0
            batch = 0

            for batch_data in tqdm(train_loader,ncols=70,leave=False,position=3,desc=f'Epoch{epoch+1}_TRAIN'):
                if config.model_name == 'LLaVAVLS':
                    for k, v in batch_data.items():
                        if isinstance(v, torch.Tensor):
                            batch_data[k] = v.to(config.device)
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                             batch_data[k] = [x.to(config.device) for x in v]
                    
                    output = model(batch_data)
                    gtscore = torch.vstack(batch_data['gtscore']).to(config.device)
                    loss = criterion(output, gtscore)
                else:
                    feature,gtscore,dataset_name,video_num = batch_data
                    feature = feature.to(config.device)
                    gtscore = gtscore.to(config.device)
                    output = model(feature)

                    loss = criterion(output,gtscore) 
                
                loss.requires_grad_(True)
                
                update_loss += loss
                batch += 1

                if batch==batch_size:
                    optimizer.zero_grad()
                    update_loss = update_loss / batch
                    update_loss.backward()
                    optimizer.step()
                    update_loss = 0.0
                    batch = 0

            if batch>0:
                optimizer.zero_grad()
                update_loss = update_loss / batch
                update_loss.backward()
                optimizer.step()
                update_loss = 0.0
                batch = 0

            val_spears = []
            val_kendalls = []
            model.eval()
            with torch.no_grad():
                for batch_data in tqdm(test_loader,ncols=70,leave=False,position=3,desc=f'Epoch{epoch+1}_TEST'):
                    if config.model_name == 'LLaVAVLS':
                        for k, v in batch_data.items():
                            if isinstance(v, torch.Tensor):
                                batch_data[k] = v.to(config.device)
                            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                                batch_data[k] = [x.to(config.device) for x in v]
                        output = model(batch_data)
                        dataset_name = dataset
                        video_num = batch_data['video_name'][0]
                    else:
                        feature,gtscore,dataset_name,video_num = batch_data
                        feature = feature.to(config.device)
                        gtscore = gtscore.to(config.device)
                        output = model(feature)

                    if dataset_name in ['SumMe','TVSum']:
                        with h5py.File(f'./data/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5','r') as hdf:
                            user_summary = np.array(hdf[video_num]['user_summary'])
                            sb = np.array(hdf[f"{video_num}/change_points"])
                            n_frames = np.array(hdf[f"{video_num}/n_frames"])
                            positions = np.array(hdf[f"{video_num}/picks"])
                        scores = output.squeeze().clone().detach().cpu().numpy().tolist()
                        summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
                        if dataset_name=='SumMe':
                            spear,kendall = get_corr_coeff([summary],[video_num],dataset_name,user_summary)
                        elif dataset_name=='TVSum':
                            spear,kendall = get_corr_coeff([scores],[video_num],dataset_name,user_scores)
                        
                        val_spears.append(spear)
                        val_kendalls.append(kendall)

            if np.mean(val_kendalls) > model_selection_kendall and np.mean(val_spears) > model_selection_spear:
                model_selection_kendall = np.mean(val_kendalls)
                model_selection_spear = np.mean(val_spears)
                torch.save(model.state_dict(), './tmp/weight.pt')
        shutil.move('./tmp/weight.pt', f'./weights/{dataset}/split{split_id+1}.pt')
