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


class CAN2(nn.Module):
    def __init__(self, modalities, fusion_method, tcn_settings, backbone_settings, output_dim, root_dir, device, semantic_context_path, compute_att_maps, frozen_resnet50, args):
        super().__init__()
        
        self.args = args
        self.device = device
        self.fusion_method = fusion_method
        self.root_dir = root_dir
        
        self.temporal = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        
        self.spatial = self.load_visual_backbone(backbone_settings=backbone_settings, frozen_resnet50=frozen_resnet50)

        if self.fusion_method == 'proposed3':
            self.fuser = CrossAttentionFusion(dim=768)
        
        if self.fusion_method in ['proposed1', 'proposed2', 'proposed3']:
            self.sentences_embeddings_features = self.create_sentence_embeddings()
            
        
        if self.fusion_method in ['proposed2_orthogonal']:
            self.sentences_embeddings_features = self.create_sentence_embeddings()
            self.sentences_embeddings_features = self.sentences_embeddings_features.cpu().numpy()
            Q, R = np.linalg.qr(self.sentences_embeddings_features.T)
            X_orthogonal = torch.tensor(Q.T).to(self.device)
            self.sentences_embeddings_features = X_orthogonal
            
        
        for modal in modalities:
            self.temporal[modal] = TemporalConvNet(num_inputs=tcn_settings[modal]['input_dim'],
                                                   num_channels=tcn_settings[modal]['channel'],
                                                   kernel_size=tcn_settings[modal]['kernel_size'])
            self.bn[modal] = BatchNorm1d(tcn_settings[modal]['channel'][-1] )
        
        # Video and context modalities
        if 'video' in modalities and 'context' in modalities:
            self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(0.4),
                                       Flatten(),
                                       Linear(512 * 5 * 5, 512),
                                       BatchNorm1d(512))
            
            if self.args.context_feature_model in ['bert', 'bert_blurred']:
                self.context_fc1 = Linear(768, 512)
                if self.fusion_method == 'proposed2':
                    self.context_fc1 = Linear(768*2, 512)
                self.context_bn1 = BatchNorm1d(512)
            elif self.args.context_feature_model in ['llama2', 'llama2_blurred','qwen3', 'qwen3_blurred']:
                self.context_fc1 = Linear(4096, 512)
                if self.fusion_method == 'proposed2':
                    self.context_fc1 = Linear(4096*2, 512)
                self.context_bn1 = BatchNorm1d(512)
                
            self.fc1 = Linear(1024, 512)
            self.bn1 = BatchNorm1d(512)
            self.fc2 = Linear(512, 8)
            
        # Video modality only
        elif 'video' in modalities:
            
            
            self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(0.4),
                                       Flatten(),
                                       Linear(512 * 5 * 5, 512),
                                       BatchNorm1d(512))
            
            self.video_fc1 = Linear(16* len(modalities), 16* len(modalities))
            self.video_bn1 = BatchNorm1d(16 * len(modalities))
            self.video_fc2 = Linear(16* len(modalities), 8)
        
        # context modality only
        elif 'context' in modalities:
            
            if self.args.context_feature_model in ['bert', 'bert_blurred']:
                self.context_fc1 = Linear(768, 512)
                if self.fusion_method == 'proposed2':
                    self.context_fc1 = Linear(768*2, 512)
                self.context_bn1 = BatchNorm1d(512)
                self.context_fc2 = Linear(512, 8)
            elif self.args.context_feature_model in ['llama2', 'llama2_blurred','qwen3', 'qwen3_blurred']:
                self.context_fc1 = Linear(4096, 512)
                if self.fusion_method == 'proposed2':
                    self.context_fc1 = Linear(4096*2, 512)
                self.context_bn1 = BatchNorm1d(512)
                self.context_fc2 = Linear(512, 8)
                
            else:
                raise ValueError(f"Context feature model {self.args.context_feature_model} not recognized.")
            
    def load_visual_backbone(self, backbone_settings, frozen_resnet50):
        resnet = VisualBackbone2(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['visual_state_dict'] + ".pth"),map_location='cpu')
        resnet.load_state_dict(state_dict)
        for param in resnet.parameters():
            param.requires_grad = not frozen_resnet50
            
        return resnet
    

    
    
    def forward(self, X):

        if 'video' in X:
            x_video = X['video']

            batch_size, length, channel, width, height = x_video.shape
            x_video = x_video.view(-1, channel, width, height) # [batch x length, channel, width, height]
            x_video = self.spatial(x_video) # [batch x length, features, width, height]
            _,feature_dim , _, _  = x_video.shape
        
            x_video = self.output_layer(x_video)
            # x_video = l2_norm(x_video)
            x_video = x_video.view(batch_size, length, feature_dim).permute(0,2,1)
        
        if 'context' in self.args.modality and 'video' in self.args.modality and self.fusion_method == 'proposed1':
            x_context = X['context']
            x_context = x_context.view(x_context.shape[0]*x_context.shape[1], x_context.shape[2])  # Flatten context features
            sentences_embeddings = self.sentences_embeddings_features

            
            similarities = cos_sim(x_context, sentences_embeddings)
            similarities = torch.argmax(similarities, dim=1)
            
            selected_embeddings = sentences_embeddings[similarities]
            
            x_context = selected_embeddings.view(X['context'].shape[0], X['context'].shape[1], X['context'].shape[2])
            
            x_context = self.context_fc1(x_context).transpose(1, 2)
            x_context = self.context_bn1(x_context)
            x_context = F.leaky_relu(x_context)
            
            x_concat = torch.cat((x_video, x_context), dim=1)

            x_concat = self.fc1(x_concat.transpose(1, 2)).transpose(1, 2)
            x_concat = self.bn1(x_concat).transpose(1, 2)       
            
            x_concat = F.leaky_relu(x_concat)
            x_concat = self.fc2(x_concat)
            
            return x_concat
        
        elif 'context' in self.args.modality and 'video' in self.args.modality and self.fusion_method == 'proposed2':
            x_context = X['context']
            x_context = x_context.view(x_context.shape[0]*x_context.shape[1], x_context.shape[2])  # Flatten context features
            sentences_embeddings = self.sentences_embeddings_features

            
            similarities = cos_sim(x_context, sentences_embeddings)
            similarities = torch.argmax(similarities, dim=1)
            
            selected_embeddings = sentences_embeddings[similarities]
            
            x_context = selected_embeddings.view(X['context'].shape[0], X['context'].shape[1], X['context'].shape[2])
            
            x_context = torch.cat((x_context, X['context']), dim=2)
            
            x_context = self.context_fc1(x_context).transpose(1, 2)
            x_context = self.context_bn1(x_context)
            x_context = F.leaky_relu(x_context)
            
            x_concat = torch.cat((x_video, x_context), dim=1)

            x_concat = self.fc1(x_concat.transpose(1, 2)).transpose(1, 2)
            x_concat = self.bn1(x_concat).transpose(1, 2)       
            
            x_concat = F.leaky_relu(x_concat)
            x_concat = self.fc2(x_concat)
            
            return x_concat
        
        elif 'context' in self.args.modality and 'video' in self.args.modality and self.fusion_method == 'proposed3':
            x_context = X['context']
            x_context = x_context.view(x_context.shape[0]*x_context.shape[1], x_context.shape[2])  # Flatten context features
            sentences_embeddings = self.sentences_embeddings_features

            
            similarities = cos_sim(x_context, sentences_embeddings)
            similarities = torch.argmax(similarities, dim=1)
            
            selected_embeddings = sentences_embeddings[similarities]
            
            x_context = selected_embeddings.view(X['context'].shape[0], X['context'].shape[1], X['context'].shape[2])
            
            # FUSION
            # x_context = torch.cat((x_context, X['context']), dim=2)
            x_context = self.fuser(x_context, X['context'])
            
            x_context = self.context_fc1(x_context).transpose(1, 2)
            x_context = self.context_bn1(x_context)
            x_context = F.leaky_relu(x_context)
            
            x_concat = torch.cat((x_video, x_context), dim=1)

            x_concat = self.fc1(x_concat.transpose(1, 2)).transpose(1, 2)
            x_concat = self.bn1(x_concat).transpose(1, 2)       
            
            x_concat = F.leaky_relu(x_concat)
            x_concat = self.fc2(x_concat)
            
            return x_concat
        
        elif 'context' in self.args.modality and 'video' in self.args.modality:
            x_context = X['context']
            x_context = self.context_fc1(x_context).transpose(1, 2)
            x_context = self.context_bn1(x_context)
            x_context = F.leaky_relu(x_context)
            
            x_concat = torch.cat((x_video, x_context), dim=1)

            x_concat = self.fc1(x_concat.transpose(1, 2)).transpose(1, 2)
            x_concat = self.bn1(x_concat).transpose(1, 2)       
            
            x_concat = F.leaky_relu(x_concat)
            x_concat = self.fc2(x_concat)
            
            return x_concat
        
        elif 'context' in self.args.modality:
            x_context = X['context']
            x_context = self.context_fc1(x_context).transpose(1, 2)
            x_context = self.context_bn1(x_context)    
            x_context = F.leaky_relu(x_context)
            x_context = self.context_fc2(x_context.transpose(1, 2))
            
            return x_context 
        
        elif 'video' in self.args.modality:
            x_video = self.temporal['video'](x_video)
            x_video = self.bn['video'](x_video)
            
            x_video = self.video_fc1(x_video.transpose(1, 2)).transpose(1, 2)
            x_video = self.video_bn1(x_video).transpose(1, 2)       
            
            x_video = F.leaky_relu(x_video)
            x_video = self.video_fc2(x_video)
            
            return x_video
        
        
        elif 'video' in self.args.modality:
            x_video = self.temporal['video'](x_video)
            x_video = self.bn['video'](x_video)
            
            x_video = self.video_fc1(x_video.transpose(1, 2)).transpose(1, 2)
            x_video = self.video_bn1(x_video).transpose(1, 2)       
            
            x_video = F.leaky_relu(x_video)
            x_video = self.video_fc2(x_video)
            
            return x_video
        
        else:
            raise ValueError('Modality not recognized.')
            
        
    
    def create_sentence_embeddings(self, semantic_context_path='/home/ens/AS84330/Context/ABAW3_EXPR4/assets/semantic_context_background_long.json'):
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        if self.args.context_feature_model == 'bert':
            model = SentenceTransformer("bert-base-nli-mean-tokens", device=self.device)
        elif self.args.context_feature_model == 'qwen3':
            model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=self.device)
        elif self.args.context_feature_model == 'llama2':
            model_id = "meta-llama/Llama-2-7b-hf"
            # Login first with: huggingface-cli login
            token = "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"
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModel.from_pretrained(model_id, output_hidden_states=True, token=token)
            model.to(self.device) 
        else:
            raise ValueError("Invalid model")
            
        model.eval()
        
        sentences = []
        with open(semantic_context_path, 'r') as file:
            contextual_info = json.load(file)
        for emotion in contextual_info['emotions']: 
            sentences.append(emotion['description'])

        sentences_label =[]
        for emotion in contextual_info['emotions']: 
            sentences_label.append(emotion['name'])
        
        
        if self.args.context_feature_model == 'llama2':
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get hidden states from the model
            with torch.no_grad():
                outputs = model(**inputs)

            # Use the last hidden state
            last_hidden_state = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]

            # Option 1: Use [first token] embedding (like CLS)
            # cls_embedding = last_hidden_state[:, 0, :]  # shape: [1, hidden_dim]

            # Option 2: Mean pooling over all tokens
            mean_embedding = last_hidden_state.mean(dim=1)  # shape: [1, hidden_dim]
            sentence_embeddings = mean_embedding.cpu().numpy() 
            
        
        else: 
            sentence_embeddings = model.encode(sentences)
        
        # embedding_matrix = nn.Embedding(8, 768)
        # embedding_matrix.weight.data.copy_(torch.from_numpy(sentence_embeddings).to(self.device)).to(self.device)

        return torch.tensor(sentence_embeddings).to(self.device)  #
    

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
        
        self.mlp = Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 8)
        )
        
        self.mlp2 = Sequential(
            # nn.Dropout(p=0.3),
            nn.Linear(128, 8)
        )
       

    def forward(self, X, use_extracted_feats):

            if use_extracted_feats:
                # clip_feats = X['clip_feats']
                # clip_feats = clip_feats.float().permute(0,2,1)
                
                # x_video = self.temporal(clip_feats)
                
                # x_video = x_video.permute(0,2,1).reshape(-1, 128)
                # x_video = self.mlp(x_video)
                # return x_video, None
                
                clip_feats = X['clip_feats']
                clip_feats = clip_feats.squeeze(1).permute(0,2,1).float()
                clip_feats = self.temporal(clip_feats)
                clip_feats = clip_feats.permute(0,2,1)
                x_video = self.mlp2(clip_feats)
                return x_video, clip_feats
                
            
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
            
            
            
            
        