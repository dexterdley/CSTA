import h5py
import numpy as np
import shutil
import torch
import torch.nn as nn
import warnings
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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

# Suppress warning regarding tensor creation from list of numpy arrays
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")

# Load configurations
config = get_config()
config.model_name = 'LLaVAVLS'
config.output_name = 'lstm'
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
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        return self.mlp_head(x)

class LLaVAVLS_PL(pl.LightningModule):
    def __init__(self, config, llava_model, tokenizer, dataset_name):
        super().__init__()
        self.config = config
        self.llava_model = llava_model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

        # Freeze LLaVA model
        self.llava_model.eval()
        for param in self.llava_model.parameters():
            param.requires_grad = False

        # Trainable video summarization model
        if hasattr(self.config, 'output_name') and "lstm" in self.config.output_name:
            self.attn_model = LSTMAttnModel(self.llava_model.config.hidden_size)
        else:
            self.attn_model = AttnModel(self.llava_model.config.hidden_size)
            
        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []

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

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        if self.tokenizer.pad_token_id is None:
                if "qwen" in self.tokenizer.name_or_path.lower():
                    # print("Setting pad token to bos token for qwen model.")
                    self.tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long()

        all_hidden_states = []
        chunk_size = 32
        for i in range(0, vid_len, chunk_size):
            chunk_video = video[i:i+chunk_size]
            chunk_len = chunk_video.shape[0]
            chunk_input_ids = input_ids.repeat(chunk_len, 1)
            chunk_attention_masks = attention_masks.repeat(chunk_len, 1)

            with torch.no_grad():
                logits, _, hidden_states = self.llava_model(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_masks,
                    images=chunk_video,
                    image_sizes=[[item] * getattr(self.config, 'clip_length', 32) for item in video_sizes],
                    modalities=['image'] * chunk_len,
                    dpo_forward=True
                )
            all_hidden_states.append(hidden_states[:, -1].float().cpu())
            del logits, hidden_states, chunk_video, chunk_input_ids, chunk_attention_masks
            torch.cuda.empty_cache()

        last_hidden_states = torch.cat(all_hidden_states, dim=0).to(self.device)
        
        impt_scores = self.attn_model(last_hidden_states)
        return impt_scores

    def training_step(self, batch, batch_idx):
        if isinstance(batch['gtscore'], list):
             gtscore = torch.vstack([g.to(self.device) for g in batch['gtscore']])
        else:
             gtscore = batch['gtscore']
             
        output = self(batch)
        loss = self.criterion(output, gtscore)
        self.log('train_loss', loss, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        video_name = batch['video_name'][0]
        self.validation_step_outputs.append({
            'output': output.detach().cpu(),
            'video_name': video_name
        })
        return None

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        val_spears = []
        val_kendalls = []
        user_scores = get_gt(self.dataset_name)

        for item in self.validation_step_outputs:
            output = item['output']
            video_num = item['video_name']
            
            with h5py.File(f'./data/eccv16_dataset_{self.dataset_name.lower()}_google_pool5.h5','r') as hdf:
                user_summary = np.array(hdf[video_num]['user_summary'])
                sb = np.array(hdf[f"{video_num}/change_points"])
                n_frames = np.array(hdf[f"{video_num}/n_frames"])
                positions = np.array(hdf[f"{video_num}/picks"])
            
            scores = output.squeeze().clone().detach().numpy().tolist()
            summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
            
            if self.dataset_name=='SumMe':
                spear,kendall = get_corr_coeff([summary],[video_num],self.dataset_name,user_summary)
            elif self.dataset_name=='TVSum':
                spear,kendall = get_corr_coeff([scores],[video_num],self.dataset_name,user_scores)
            
            val_spears.append(spear)
            val_kendalls.append(kendall)
            
        avg_kendall = np.mean(val_kendalls)
        avg_spear = np.mean(val_spears)
        
        self.log('val_kendall', avg_kendall, sync_dist=True)
        self.log('val_spearman', avg_spear, sync_dist=True)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.config.learning_rate), weight_decay=float(self.config.weight_decay))

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
            for i in range(1):
                if dataset == 'SumMe':
                    DatasetCls = SumMeLLaMA_VideoDataset
                    TrainCollator = SumMeTrainCollator
                    ValCollator = SumMeValCollator
                elif dataset == 'TVSum':
                    DatasetCls = TVSumLLaMA_VideoDataset
                    TrainCollator = TVSumTrainCollator
                    ValCollator = TVSumValCollator
                
                clip_length = getattr(config, 'clip_length', 32)
                
                train_ds = DatasetCls(mode='train', split_idx=i, image_processor=image_processor, clip_length=clip_length, hidden_size=512, load_test=False)
                test_ds = DatasetCls(mode='test', split_idx=i, image_processor=image_processor, clip_length=clip_length, hidden_size=512, load_test=False)
                
                train_l = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=TrainCollator(), pin_memory=True)
                test_ds = DatasetCls(mode='test', split_idx=i, image_processor=image_processor, clip_length=clip_length, hidden_size=512, load_test=True)
                test_l = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=ValCollator(mode="val"), pin_memory=True)
                yield i, (train_l, test_l)
        dataloader_iterator = create_llava_splits()
    else:
        dataloader_iterator = enumerate(create_dataloader(dataset))

    for split_id,(train_loader, test_loader) in tqdm(dataloader_iterator,total=1,ncols=70,leave=False,position=1,desc=dataset):
        if config.model_name == 'LLaVAVLS':
            model = LLaVAVLS_PL(config, llava_model, tokenizer, dataset)
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=f'./weights/{dataset}/',
                filename=f'split{split_id+1}',
                monitor='val_kendall',
                mode='max',
                save_top_k=1
            )

            logger = TensorBoardLogger(save_dir="runs",
            default_hp_metric=False,
            name=config.model_name + "_" + config.output_name + "_" + dataset + "_" + str(split_id)
            )

            logger.experiment.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
            )
            
            trainer = pl.Trainer(
                max_epochs=config.epochs,
                accelerator='gpu',
                devices='auto',
                strategy='ddp',
                logger=logger,
                callbacks=[checkpoint_callback],
                enable_progress_bar=True,
                log_every_n_steps=1,
                precision=32,
                accumulate_grad_batches=batch_size,
                val_check_interval=1.0
            )
            
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            
            # Save the trained model weights
            save_path = f'./weights/{dataset}/split{split_id+1}_attn_model.pth'
            if checkpoint_callback.best_model_path:
                print(f"Loading best checkpoint from {checkpoint_callback.best_model_path}")
                checkpoint = torch.load(checkpoint_callback.best_model_path, map_location='cpu')
                state_dict = {k.replace('attn_model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('attn_model.')}
                torch.save(state_dict, save_path)
                print(f"Saved best model weights to {save_path}")
            else:
                torch.save(model.attn_model.state_dict(), save_path)
                print(f"Saved last model weights to {save_path}")
            
        else:
            # Legacy code for other models (not LLaVA)
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
