from models.temporal_convolutional_model import TemporalConvNet
from models.backbone import VisualBackbone2
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np 
from transformers import AutoModel, AutoTokenizer
import os
import json
import torch

from torch import nn
from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt

import lorem
import clip
from torchvision import models
from torch.nn.utils import weight_norm
from transformers import AutoImageProcessor, AutoModelForImageClassification

EMOTIONS = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, A, B):
        # A attends to B
        # Query = A, Key = Value = B
        attended, _ = self.attn(query=A, key=B, value=B)
        return self.norm(A + attended)  # optional residual + norm
    
    

class Video_only(nn.Module):
    def __init__(self, root_dir, device, backbone_settings, frozen_resnet50):
        super().__init__()
        self.root_dir = root_dir
        self.device = device
        
        self.temporal = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        
        self.spatial = self.load_visual_backbone(backbone_settings=backbone_settings, frozen_resnet50=frozen_resnet50)
        self.output_layer = Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Mean pool to [B, 512, 1, 1]
            nn.Flatten(), 
        )
        # self.classifier = Sequential(
        #     nn.Linear(512, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 8)
        # )
        embedding_dim = 512
        self.output_layer2 = Sequential(
            # BatchNorm2d(embedding_dim),
            Flatten(),
            nn.ReLU(),
            Dropout(0.4),
            Linear(embedding_dim * 5 * 5, 512),
            nn.LayerNorm(embedding_dim)
            # BatchNorm1d(embedding_dim)
        )
        
        self.classifier = Sequential(
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            Dropout(0.4),   
            nn.Linear(128, 8)
        )
        
        self.temporal = TemporalConvNet(num_inputs=512, max_length=300,
                                                   num_channels=[512,256, 128], attention=0,
                                                   kernel_size=5, dropout=0.3).to(self.device)
        
        # context modality on
    def load_visual_backbone(self, backbone_settings, frozen_resnet50):
        resnet = VisualBackbone2(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['visual_state_dict'] + ".pth"),map_location='cpu')
        resnet.load_state_dict(state_dict)
        for param in resnet.parameters():
            param.requires_grad = not frozen_resnet50
            
        return resnet

    def forward(self, X, features):
            
            batch_size, length, channel, width, height = X['video'].shape
            video_features = None 
            
            if features is not None:
                x_video = features.to(self.device).to(torch.float32)
                x_video = x_video.view(-1, 512, 5, 5) 
                x_video = self.output_layer2(x_video)
                x_video = x_video.view(batch_size,length, 512).permute(0,2,1)
                x_video = self.temporal(x_video)
                x_video = x_video.permute(0,2,1).reshape(batch_size*length, 128)
                x_video = self.classifier(x_video)
                
            else: 
                x_video = X['video']
                x_video = x_video.view(-1, channel, width, height) # [batch x length, channel, width, height]
                x_video = self.spatial(x_video)
                video_features = x_video
                video_features = video_features.view(batch_size, length, 512, 5, 5)
                x_video = self.output_layer(x_video)
            

            
            x_video = x_video.view(batch_size, length, -1)  
            
            return x_video, video_features
        

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x  
    

class AttentionFusion(nn.Module):
    """ Fuse modalities using attention. """

    def __init__(self,
                 num_feats_modality: list,
                 num_out_feats: int = 256):
        """ Instantiate attention fusion instance.

        Args:
            num_feats_modality (list): Number of features per modality.
            num_out_feats (int): Number of output features.
        """

        super(AttentionFusion, self).__init__()

        self.attn = nn.ModuleList([])
        for num_feats in num_feats_modality:
            self.attn.append(
                nn.Linear(num_feats, num_out_feats))

        self.weights = nn.Linear(num_out_feats * len(num_feats_modality), num_out_feats * len(num_feats_modality))
        self.num_features = num_out_feats * len(num_feats_modality)

    def forward(self, x: list):
        """ Forward pass

        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """

        proj_m = []
        for i, m in enumerate(x.values()):
            proj_m.append(self.attn[i](m.transpose(1, 2)))

        attn_weights = F.softmax(
            self.weights(torch.cat(proj_m, -1)), dim=-1)

        out_feats = attn_weights * torch.cat(proj_m, -1)

        return out_feats

    
class Proposed(nn.Module):
    def __init__(self, root_dir, device, backbone_settings, frozen_resnet50, args):
        super().__init__()
        
        self.device = device
        self.spatial, _ = clip.load('ViT-L/14', device=device) 
        self.spatial = self.spatial.float()
        
        
        # for param in self.spatial.parameters():
        #     param.requires_grad = bool(args.unfreeze_all_clip) 
        
                
        # for param in self.spatial.visual.transformer.resblocks[-1].parameters():
        #     param.requires_grad = True
        # for param in self.spatial.visual.ln_post.parameters():
        #     param.requires_grad = True
        # self.spatial.visual.proj.requires_grad = True 
        
        # self.processor = AutoImageProcessor.from_pretrained("mo-thecreator/vit-Facial-Expression-Recognition")
        # self.spatial = AutoModelForImageClassification.from_pretrained("mo-thecreator/vit-Facial-Expression-Recognition", output_hidden_states=True)
        
        # self.spatial.to(self.device)
        # self.spatial = self.spatial.float()
        
        #self.spatial= ResNet18Backbone().to(device) 
        tcn_channels = [768, 256, 128]
        self.temporal = TemporalConvNet(num_inputs=tcn_channels[0],
                                        num_channels=tcn_channels,
                                        dropout=0.3,
                                        kernel_size=5, 
                                        attention=False,
                                        ).to(self.device)
        
        tcn_channels = [4096, 512, 128]
        self.temporal_context = TemporalConvNet(num_inputs=tcn_channels[0],
                                        num_channels=tcn_channels,
                                        dropout=0.3,
                                        kernel_size=5, 
                                        attention=False,
                                        ).to(self.device)
        
        # tcn_channels = [128, 128, 128]
        # self.temporal_audio = TemporalConvNet(num_inputs=tcn_channels[0],
        #                                 num_channels=tcn_channels,
        #                                 dropout=0.3,
        #                                 kernel_size=5, 
        #                                 attention=False,
        #                                 ).to(self.device)
        
        self.mlp = Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 8)
        )
        
        # self.log_temperature = nn.Parameter(torch.tensor(0.0))
        # self.log_temperature_all_emotions = nn.Parameter(torch.zeros(8))
        
        
        self.mlp_vis = Sequential(
            # nn.Dropout(p=0.3),
            nn.Linear(128, 8)
        )
        
        self.mlp_context = Sequential(
            # nn.Dropout(p=0.3),
            nn.Linear(128, 8)
        )
        
        self.mlp = Sequential(
            nn.Linear(256, 8),)
        
        self.gate = nn.Parameter(torch.ones(8))
        
        self.fuse = AttentionFusion([128], num_out_feats=128)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc1 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 8)
        
       

    def forward(self, X, use_extracted_feats):

            if use_extracted_feats:
                # clip_feats = X['clip_feats']
                # clip_feats = clip_feats.float().permute(0,2,1)
                
                # x_video = self.temporal(clip_feats)
                
                # x_video = x_video.permute(0,2,1).reshape(-1, 128)
                # x_video = self.mlp(x_video)
                # return x_video, None
                
                context_feats = X['context']
                context_feats = context_feats.squeeze(1).permute(0,2,1)
                
                clip_feats = X['clip_feats']
                clip_feats = clip_feats.squeeze(1).permute(0,2,1).float()
                
                # audio_feats = X['vggish']
                # audio_feats = audio_feats.squeeze(1).permute(0,2,1).float()    
                
                # Temporal
                clip_feats = self.temporal(clip_feats)
                clip_feats = self.bn1(clip_feats)
                context_feats = self.temporal_context(context_feats)
                context_feats = self.bn2(context_feats)
                
                # audio_feats = self.temporal_audio(audio_feats)
                
                # concat_feats = torch.cat((clip_feats, audio_feats), dim=1)
                # concat_feats = concat_feats.permute(0,2,1)
                
                x_concat = self.fuse({'video': clip_feats})
                # x_concat = self.mlp(x_concat)
                
                c = self.fc1(x_concat).transpose(1, 2)
                c = self.bn1(c).transpose(1, 2)
                c = F.leaky_relu(c)
                c = self.fc2(c)
                c = torch.tanh(c)
                #x_video = self.mlp_vis(clip_feats.permute(0,2,1))
                #x_context = self.mlp_context(context_feats.permute(0,2,1))
                
                # g = self.gate.unsqueeze(0).unsqueeze(0).expand(x_video.size(0), x_video.size(1), -1)
                # x_concat = x_video * g + X['stimuli_weights'] * (1 - g)
                # Stimuli weight multiplication
                # temperature = torch.exp(self.log_temperature)
                # # temperature = 0.5
                # stimuli_weights = X['stimuli_weights'].to(self.device)  
                # stimuli_weights = stimuli_weights.unsqueeze(1).expand(-1, x_video.size(1), -1)  
                # stimuli_weights = F.softmax(stimuli_weights / temperature, dim=-1) # Forgot softmax ? 
                # x_video = x_video * stimuli_weights
                
                # out = F.softmax(x_video, dim=2) * X['stimuli_weights']
                return c , clip_feats
                
            
            else: 
                x_video = X['video']
                batch_size, length, channel, width, height = x_video.shape
            
                x_video = x_video.view(-1, channel, width, height)
                x_video = F.interpolate(x_video, size=(224, 224), mode='bilinear', align_corners=False)# [batch x length, channel, width, height]
                
                with torch.no_grad():
                    # inputs = self.processor(images=x_video, return_tensors="pt")
                    clip_feats = self.spatial.encode_image(x_video)
                    
                # clip_feats = clip_feats.hidden_states[-1][:, 0, :]
                x_video = self.mlp2(clip_feats)
                return x_video, clip_feats
            
class Context_only(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        tcn_settings = {
            'context': {
                'input_dim': 4096,
                'channel': [4096, 512, 128],
                'kernel_size': 5 
            },
            }
        
        modalities = ['context']
        self.temporal = TemporalConvNet(num_inputs=tcn_settings['context']['input_dim'],
                                                   num_channels=tcn_settings['context']['channel'],
                                                   kernel_size=tcn_settings['context']['kernel_size'])
        self.bn = BatchNorm1d(tcn_settings['context']['channel'][-1] )
        
        feas_modalities = [tcn_settings[modal]['channel'][-1] for modal in modalities]
        self.fuse = AttentionFusion(num_feats_modality=feas_modalities, num_out_feats=128)
        
        self.bn1 = BatchNorm1d(128 * len(modalities))
        self.fc1 = Linear(128* len(modalities), 128* len(modalities))
        self.fc2 = Linear(128* len(modalities), 8)
        
    def forward(self, context_feats):
        x = context_feats.squeeze(1).transpose(1, 2)
        x = self.temporal(x.float())
        x = self.bn(x)
        
        c = self.fuse({'context': x})
        c = self.fc1(c).transpose(1, 2)
        c = self.bn1(c).transpose(1, 2)
        c = F.leaky_relu(c)
        c = self.fc2(c)
        c = torch.tanh(c)
        
        return c
class CAN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        modalities = ['clip_feats','vggish']
        tcn_settings = {
            'clip_feats': {
                'input_dim': 768,
                'channel': [768, 256, 128],
                'kernel_size': 5
            },
            'context': {
                'input_dim': 4096,
                'channel': [4096, 512, 128],
                'kernel_size': 5 
            },
            'vggish': {
                'input_dim': 128,
                'channel': [128, 128, 128],
                'kernel_size': 5
                }
            }


        self.temporal = nn.ModuleDict()
        self.up_sample = nn.ModuleDict()
        self.bn = nn.ModuleDict()

        self.spatial = nn.ModuleDict()


        for modal in modalities:
            self.temporal[modal] = TemporalConvNet(num_inputs=tcn_settings[modal]['input_dim'],
                                                   num_channels=tcn_settings[modal]['channel'],
                                                   kernel_size=tcn_settings[modal]['kernel_size'])
            self.bn[modal] = BatchNorm1d(tcn_settings[modal]['channel'][-1] )


        feas_modalities = [tcn_settings[modal]['channel'][-1] for modal in modalities]
        self.fuse = AttentionFusion(num_feats_modality=feas_modalities, num_out_feats=128)

        self.conv_c = nn.Conv1d(128 * len(modalities), 128, 1)

        self.bn1 = BatchNorm1d(128 * len(modalities))
        self.fc1 = Linear(128* len(modalities), 128* len(modalities))
        self.fc2 = Linear(128* len(modalities), 8)

        self.context_only = Context_only(device=device)
    


    def forward(self, modalities, use_extracted_feats, train_mode, proposed):

        X = {}
        X['clip_feats'] = modalities['clip_feats']
        X['vggish'] = modalities['vggish']
        x = {}

        for modal in X:
            x[modal] = X[modal].squeeze(1).transpose(1, 2)
            x[modal] = self.temporal[modal](x[modal].float())
            x[modal] = self.bn[modal](x[modal])

        c = self.fuse(x)
        c = self.fc1(c).transpose(1, 2)
        c = self.bn1(c).transpose(1, 2)
        c = F.leaky_relu(c)
        c = self.fc2(c)
        c = torch.tanh(c)

        # if train_mode == False and proposed:
            # TEST ONE 
        # if train_mode == False and proposed:
        #     visualise = c.detach().cpu().numpy()
        #     for i, batch in enumerate(visualise):
        #         for k, length in enumerate(batch):
        #             two = np.where(length > 0.6)[0]
        #             if len(two) > 1:
        #                 max_indice = np.argmax(modalities['stimuli_weights'][i,k,:][two].detach().cpu().numpy())
        #                 c[i,k,two[max_indice]] = 1
            
            
            # TEST TWO
            # contextual = self.context_only(modalities['context']) 
            # visualise = c.detach().cpu().numpy()
            # for i, batch in enumerate(visualise):
            #     for k, length in enumerate(batch):
            #         two = np.where(length > 0.4)[0]
            #         if len(two) > 1:
            #             #max_indice = np.argmax(contextual[i,k,:][two].detach().cpu().numpy())
            #             max_indice = np.random.randint(0, len(two))
            #             c[i,k,two[max_indice]] = 1 
        
        return c , None