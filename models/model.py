from models.arcface_model import Backbone
from models.temporal_convolutional_model import TemporalConvNet
from models.transformer import MultimodalTransformerEncoder, IntraModalTransformerEncoder, InterModalTransformerEncoder
from models.backbone import VisualBackbone, AudioBackbone, ContextBackbone, VisualBackbone2
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

import json

import math
import os
import torch
from torch import nn

import numpy as np
from umap import UMAP


from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module
import torch.nn.functional as F

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class my_res50(nn.Module):
    def __init__(self, input_channels=3, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir",
                 embedding_dim=512):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')

            if "backbone" in list(state_dict.keys())[0]:

                self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                        Dropout(0.4),
                                                        Flatten(),
                                                        Linear(embedding_dim * 5 * 5, embedding_dim),
                                                        BatchNorm1d(embedding_dim))

                new_state_dict = {}
                for key, value in state_dict.items():

                    if "logits" not in key:
                        new_key = key[9:]
                        new_state_dict[new_key] = value

                self.backbone.load_state_dict(new_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x, extract_cnn=False):
        x = self.backbone(x)

        if extract_cnn:
            return x

        x = self.logits(x)
        return x


class LeaderFollowerAttentionNetwork(nn.Module):
    def __init__(self, backbone_state_dict, modality=['frame'], kernel_size=5, example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128], 'cnn_res50': [512, 256, 256, 128], 'mfcc':[32, 32, 32, 32], 'vggish': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512, 'mfcc': 39, 'vggish': 128, 'egemaps': 23},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128, 'mfcc': 32, 'vggish': 32, 'egemaps': 32}, root_dir='', device='cuda'):
        super().__init__()
        self.backbone_state_dict = backbone_state_dict
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.encoderQ, self.encoderK, self.encoderV = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()

        self.final_dim = self.encoder_dim[self.modality[0]] + 32*len(self.modality)
        self.spatial = None

    def init(self):
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

            self.encoderQ[modal] = nn.Linear(self.encoder_dim[modal], 32)
            self.encoderK[modal] = nn.Linear(self.encoder_dim[modal], 32)
            self.encoderV[modal] = nn.Linear(self.encoder_dim[modal], 32)

        self.ln = nn.LayerNorm([len(self.modality), 32])
        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()
            x[modal] = x[modal].contiguous().view(batch_size * self.example_length, -1)

        Q = [self.encoderQ[modal](x[modal]) for modal in self.modality]
        K = [self.encoderK[modal](x[modal]) for modal in self.modality]
        V = [self.encoderV[modal](x[modal]) for modal in self.modality]

        Q = torch.stack(Q, dim=-2)
        K = torch.stack(K, dim=-2)
        V = torch.stack(V, dim=-2)

        QT = Q.permute(0, 2, 1)
        scores = torch.matmul(K, QT) / math.sqrt(32)
        scores = nn.functional.softmax(scores, dim=-1)

        follower = torch.matmul(scores, V)
        follower = self.ln(follower + V)
        follower = follower.view(follower.size()[0], -1)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

        return x


class LeaderFollowerAttentionNetworkWithMultiHead(nn.Module):
    def __init__(self, backbone_state_dict, modality=['frame'], kernel_size=5, example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128], 'cnn_res50': [512, 256, 256, 128], 'mfcc':[32, 32, 32, 32], 'vggish': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512, 'mfcc': 39, 'vggish': 128, 'egemaps': 23},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128, 'mfcc': 32, 'vggish': 32, 'egemaps': 32},
                 modal_dim=32, num_heads=2,
                 root_dir='', device='cuda'):
        super().__init__()
        self.backbone_state_dict = backbone_state_dict
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.fusion = nn.ModuleDict(), None
        self.num_heads = num_heads
        self.modal_dim = modal_dim
        self.final_dim = self.encoder_dim[self.modality[0]] + self.modal_dim*len(self.modality)
        self.spatial = None

    def init(self):
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()

        follower = self.fusion(x)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

        return x


class LeaderFollowerAttentionNetworkWithMultiHead(nn.Module):
    def __init__(self, backbone_state_dict, modality=['frame'], kernel_size=5, example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128], 'cnn_res50': [512, 256, 256, 128], 'mfcc':[32, 32, 32, 32], 'vggish': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512, 'mfcc': 39, 'vggish': 128, 'egemaps': 23},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128, 'mfcc': 32, 'vggish': 32, 'egemaps': 32},
                 modal_dim=32, num_heads=2,
                 root_dir='', device='cuda'):
        super().__init__()
        self.backbone_state_dict = backbone_state_dict
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.fusion = nn.ModuleDict(), None
        self.num_heads = num_heads
        self.modal_dim = modal_dim
        self.final_dim = self.encoder_dim[self.modality[0]] + self.modal_dim*len(self.modality)
        self.spatial = None

    def init(self):
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()

        follower = self.fusion(x)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

        return x


class LFAN(nn.Module):
    def __init__(self, backbone_settings, modality=['frame'], kernel_size=5, example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128], 'cnn_res50': [512, 256, 256, 128], 'mfcc':[32, 32, 32, 32], 'vggish': [32, 32, 32, 32], 'logmel': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512, 'mfcc': 39, 'vggish': 128, 'logmel': 128, 'egemaps': 88},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128, 'mfcc': 32, 'vggish': 32, 'logmel': 32, 'egemaps': 32},
                 modal_dim=32, num_heads=2,
                 root_dir='', device='cuda'):
        super().__init__()
        self.backbone_settings = backbone_settings
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.fusion = nn.ModuleDict(), None
        self.num_heads = num_heads
        self.modal_dim = modal_dim
        self.final_dim = self.encoder_dim[self.modality[0]] + self.modal_dim*len(self.modality)
        self.spatial = nn.ModuleDict()
        self.bn = nn.ModuleDict()


    def load_visual_backbone(self, backbone_settings):

        resnet = VisualBackbone(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['visual_state_dict'] + ".pth"),
                                map_location='cpu')
        resnet.load_state_dict(state_dict)

        for param in resnet.parameters():
            param.requires_grad = False

        return resnet

    def load_audio_backbone(self, backbone_settings):

        vggish = AudioBackbone()
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['audio_state_dict'] + ".pth"),
                                map_location='cpu')
        vggish.backbone.load_state_dict(state_dict)

        for param in vggish.parameters():
            param.requires_grad = False


        return vggish

    def init(self):
        self.output_dim = 1


        if 'video' in self.modality:
            self.root_dir = self.root_dir
            self.spatial["visual"] = self.load_visual_backbone(backbone_settings=self.backbone_settings)

        if 'logmel' in self.modality:
            self.root_dir = self.root_dir
            self.spatial["audio"] = self.load_audio_backbone(backbone_settings=self.backbone_settings)

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)
            self.bn[modal] = BatchNorm1d(self.tcn_channel[modal][-1])

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, X):

        if 'video' in X:
            batch_size, length, channel, width, height = X['video'].shape
            X['video'] = X['video'].view(-1, channel, width, height) # [batch x length, channel, width, height]
            X['video'] = self.spatial.visual(X['video'])
            _, feature_dim = X['video'].shape
            X['video'] = X['video'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        if 'logmel' in X:
            batch_size, height, length, width = X['logmel'].shape
            X['logmel'] = X['logmel'].permute((0, 2, 3, 1)).contiguous()
            X['logmel'] = X['logmel'].view(-1, width, height) # [batch x length, channel, width, height]
            X['logmel'] = self.spatial.audio(X['logmel'])
            _, feature_dim = X['logmel'].shape
            X['logmel'] = X['logmel'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        for modal in X:
            X[modal] = X[modal].squeeze(1).transpose(1, 2)
            X[modal] = self.temporal[modal](X[modal])
            X[modal] = self.bn[modal](X[modal]).transpose(1, 2)

        follower = self.fusion(X)

        X = torch.cat((X[self.modality[0]], follower), dim=-1)
        X = self.regressor(X)
        X = X.view(batch_size, self.example_length, -1)
        X = torch.tanh(X)
        return X


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
    
class ConcatFusion(nn.Module):
    """ Fuse modalities using feature concatenation. """

    def __init__(self,
                 num_feats_modality: list,
                 num_out_feats: int = 256):
        """ Instantiate feature concatenation fusion instance.

        Args:
            num_feats_modality (list): Number of features per modality.
            num_out_feats (int): Number of output features.
        """

        super(ConcatFusion, self).__init__()


        self.weights = nn.Linear(num_out_feats * len(num_feats_modality), num_out_feats * len(num_feats_modality))
        self.num_features = num_out_feats * len(num_feats_modality)
        
        self.fc1 = nn.Linear(sum(num_feats_modality),  self.num_features)
        self.relu = nn.ReLU()
        

    def forward(self, x: list):
        """ Forward pass

        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """

        x = torch.cat([x[modal] for modal in x], dim=1)
        x = x.transpose(1, 2)   
        x = self.fc1(x)
        x = self.relu(x)
        return x


class CAN(nn.Module):
    def __init__(self, modalities, fusion_method, tcn_settings, backbone_settings, output_dim, root_dir, device, frozen_resnet50):
        super().__init__()
        self.device = device
        self.fusion_method = fusion_method


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
        self.fuse_attention = AttentionFusion(num_feats_modality=feas_modalities, num_out_feats=128)
        self.fuse_concat = ConcatFusion(num_feats_modality=feas_modalities, num_out_feats=128)

        self.conv_c = nn.Conv1d(128 * len(modalities), 128, 1)

        self.bn1 = BatchNorm1d(128 * len(modalities))
        self.fc1 = Linear(128* len(modalities), 128* len(modalities))
        self.fc2 = Linear(128* len(modalities), 8)

        self.bn_feature_context = BatchNorm1d(128)
        self.fc_feature_context = Linear(3584, 128)
        self.dropout = nn.Dropout(p=0.6)
        
        if 'video' in modalities:
            self.root_dir = root_dir
            self.spatial["visual"] = self.load_visual_backbone(backbone_settings=backbone_settings, frozen_resnet50=frozen_resnet50)

        if 'logmel' in modalities:
            self.root_dir = root_dir
            self.spatial["audio"] = self.load_audio_backbone(backbone_settings=backbone_settings)
            
        if 'context' in modalities:
            self.root_dir = root_dir
            self.spatial["context"] = self.load_context_backbone()

    def load_visual_backbone(self, backbone_settings, frozen_resnet50):

        resnet = VisualBackbone(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['visual_state_dict'] + ".pth"),
                                map_location='cpu')
        resnet.load_state_dict(state_dict)

        for param in resnet.parameters():
            param.requires_grad = not frozen_resnet50


        return resnet
    
    def load_context_backbone(self):
        roberta = ContextBackbone()
        
        for param in roberta.parameters():
            param.requires_grad = True


        return roberta

    def load_audio_backbone(self, backbone_settings):

        vggish = AudioBackbone()
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['audio_state_dict'] + ".pth"),
                                map_location='cpu')
        vggish.backbone.load_state_dict(state_dict)

        for param in vggish.parameters():
            param.requires_grad = False


        return vggish


    def forward(self, X):

        x = {}

        if 'video' in X:
            batch_size, length, channel, width, height = X['video'].shape
            X['video'] = X['video'].view(-1, channel, width, height) # [batch x length, channel, width, height]
            X['video'] = self.spatial.visual(X['video'])
            _, feature_dim = X['video'].shape
            X['video'] = X['video'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        if 'logmel' in X:
            batch_size, height, length, width = X['logmel'].shape
            X['logmel'] = X['logmel'].permute((0, 2, 3, 1)).contiguous()
            X['logmel'] = X['logmel'].view(-1, width, height) # [batch x length, channel, width, height]
            X['logmel'] = self.spatial.audio(X['logmel'])
            _, feature_dim = X['logmel'].shape
            X['logmel'] = X['logmel'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]
            
        if 'context' in X:
            X['context'] = self.spatial.context(X['context'])
            x['context'] = X['context'].unsqueeze(2).repeat(1,1,300)#.permute(0, 2, 1) torch.Size([16, 128, 300])
            # X['context'] = X['context'].permute(0, 2, 1).unsqueeze(1)
            # X['context'] = X['context']#.unsqueeze(1).repeat(1, 300, 1)#.permute(0, 2, 1)
        if 'feature_context' in X:
            x['feature_context'] = X['feature_context'].permute(0,2,1).repeat(1,1,300).transpose(1, 2)
            x['feature_context'] = self.fc_feature_context(x['feature_context']).transpose(1, 2)
            x['feature_context'] = self.bn_feature_context(x['feature_context'])
            x['feature_context'] = F.leaky_relu(x['feature_context'])
            x['feature_context'] = self.dropout(x['feature_context'])
            

        for modal in X:
            if modal == 'context':
                x[modal] = self.bn[modal](x[modal])
                # x[modal] = torch.nn.functional.normalize(x[modal], p=2, dim=1)
                continue
            if modal == 'feature_context':
                continue
            x[modal] = X[modal].squeeze(1).transpose(1, 2)
            x[modal] = self.temporal[modal](x[modal])
            x[modal] = self.bn[modal](x[modal])
            # x[modal] = torch.nn.functional.normalize(x[modal], p=2, dim=1)

        if self.fusion_method == 'attention':
            c = self.fuse_attention(x)
        elif self.fusion_method == 'concat':
            c = self.fuse_concat(x)
        else:
            raise ValueError('Fusion method not recognized.')

        # c = self.fc1(x['context'].transpose(1, 2)).transpose(1, 2)
        # c = self.dropout(c)
        c = self.fc1(c).transpose(1, 2)
        c = self.bn1(c).transpose(1, 2)
        
        c = F.leaky_relu(c)
        c = self.fc2(c)
        # c = torch.tanh(c)

        return c



class CAN2(nn.Module):
    def __init__(self, modalities, fusion_method, tcn_settings, backbone_settings, output_dim, root_dir, device, semantic_context_path, compute_att_maps, frozen_resnet50, args):
        super().__init__()
        
        self.args = args
        self.device = device
        self.fusion_method = fusion_method
        self.compute_att_maps = compute_att_maps

        self.temporal = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        

        self.spatial = nn.ModuleDict()


        for modal in modalities:
            self.temporal[modal] = TemporalConvNet(num_inputs=tcn_settings[modal]['input_dim'],
                                                   num_channels=tcn_settings[modal]['channel'],
                                                   kernel_size=tcn_settings[modal]['kernel_size'])
            self.bn[modal] = BatchNorm1d(tcn_settings[modal]['channel'][-1] )


        feas_modalities = [tcn_settings[modal]['channel'][-1] for modal in modalities]
        
        if self.compute_att_maps:
            self.inter_connexions = self.load_inter_connexion(feas_modalities, semantic_context_path)

        # self.bn0 = BatchNorm1d(512 * len(modalities))
        # self.fc0 = Linear(512* len(modalities), 512* len(modalities))
        self.bn1 = BatchNorm1d(16 * len(modalities))
        self.fc1 = Linear(512* len(modalities), 16* len(modalities))
        
        if self.compute_att_maps:
            self.fc2 = Linear(16* len(modalities), 1)
        else: 
            self.fc2 = Linear(16* len(modalities), 8)

        # self.bn_feature_context = BatchNorm1d(128)
        # self.fc_feature_context = Linear(3584, 128)
        self.dropout = nn.Dropout(p=0.4)
        
        self.linear1 = nn.Linear(1024, 512)
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(0.4),
                                       Flatten(),
                                       Linear(512 * 5 * 5, 512),
                                       BatchNorm1d(512))
        # self.output_layer = Sequential(
        #                                 BatchNorm2d(512),
        #                                 Dropout(0.4),
        #                                 nn.AdaptiveAvgPool2d((1, 1)),  # Reduces each 512xHxW feature map to 512x1x1
        #                                 Flatten(),
        #                                 # Linear(512, 512),
        #                                 # BatchNorm1d(512)
# )
        
        if 'context' in modalities:
            if self.args.context_feature_model in ['bert', 'bert_blurred']:
                self.context_fc1 = Linear(768, 512)
            elif self.args.context_feature_model in ['llama2', 'llama2_blurred','qwen3', 'qwen3_blurred']:
                self.context_fc1 = Linear(4096, 512)
            else:
                raise ValueError(f"Context feature model {self.args.context_feature_model} not recognized.")
            
            self.context_dropout = Dropout(0.5) 
            self.context_bn1 = BatchNorm1d(512)
        
        # Umap features
        self.feature_maps = None
        self.res50_features = None
        
        self.avg_pool = nn.AvgPool2d(kernel_size=5)
        
        if 'video' in modalities:
            self.root_dir = root_dir
            self.spatial["visual"] = self.load_visual_backbone(backbone_settings=backbone_settings, frozen_resnet50=frozen_resnet50)


    
    def load_inter_connexion(self, feas_modalities, semantic_context_path):
        
        inter_connexions = InterConnexions(device=self.device, num_feats_modality=feas_modalities, num_out_feats=128, semantic_context_path=semantic_context_path, args=self.args)
        
        for name, param in inter_connexions.named_parameters():
            param.requires_grad = True
            if name == 'sentence_embeddings.weight':
                param.requires_grad = False
        
        return inter_connexions
    
    def load_visual_backbone(self, backbone_settings, frozen_resnet50):

        resnet = VisualBackbone2(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['visual_state_dict'] + ".pth"),
                                map_location='cpu')
        resnet.load_state_dict(state_dict)

        # Change here !
        # resnet.backbone.output_layer = nn.Sequential()
        
        for param in resnet.parameters():
            param.requires_grad = not frozen_resnet50


        return resnet
    
   



    
    def compute_attention_map(self, x_video):
        # Print
        # for name, param in self.inter_connexions.inter_sentence_encoder.named_parameters():
        #     if 'weight' in name:  # Filter to only show weights
        #         print(f"Layer: {name} | Weights: {param.mean()}")
        # Print
        
        batch_size, length, channel, width, height = x_video.shape
        
        c = self.inter_connexions(batch_size)
        
        
        c = c.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, 5, 5)
        
        
        cosine_sim = F.cosine_similarity(c, x_video, dim=2)
        cosine_sim = F.relu(cosine_sim)
        
        sum = cosine_sim.sum(dim=[2, 3], keepdim=True)
        alpha = cosine_sim / torch.clamp(sum,1e-6) 
        alpha = alpha.unsqueeze(4).expand(-1,-1, -1,-1, 512).permute(0, 1, 4, 2, 3)
        return alpha
    
    def compute_attention_map_eval(self, x_video, feature_dim, batch_size, length):
        alpha = torch.zeros([batch_size*length, 512, 5, 5]).to(self.device)
        for label in [0,1,2,3,4,5,6,7]:
            labels = torch.full((batch_size, length, 1), label).to(self.device)
            alpha = alpha + self.compute_attention_map(x_video, labels, feature_dim)
        return alpha / 8

    def get_res50_features(self):
        return self.res50_features
    
    def get_feature_maps(self):
        return self.feature_maps
    
    def forward(self, X):

        if 'video' in X:
            x_video = X['video']

            batch_size, length, channel, width, height = x_video.shape
            x_video = x_video.view(-1, channel, width, height) # [batch x length, channel, width, height]
            x_video = self.spatial.visual(x_video) # [batch x length, features, width, height]
            _,feature_dim , _, _  = x_video.shape
        
        if self.compute_att_maps:
            # x_video = x_video.transpose(1, 2)
            # Extract Res50 Features
            if self.training:
                self.res50_features = self.avg_pool(x_video).squeeze().view(batch_size, length, feature_dim).permute(0,2,1)
            # INTER
            
            x_video = x_video.unsqueeze(1).expand(-1,8,-1, -1, -1) # [batch x length, class_dim, features, width, height]
            
            att_map = self.compute_attention_map(x_video)
            
            att_x_video = att_map * x_video
            # att_x_video = x_video  
            att_x_video = att_x_video.reshape(-1, 512, 5, 5) # [batch x length x class_dim, features, width, height] 
            # att_x_video = att_x_video.view(batch_size*length*8, 512, 25)
            # att_x_video = torch.sum(att_x_video, dim=2) # [batch x length x class_dim, features]
            att_x_video = self.output_layer(att_x_video)
            # att_x_video = l2_norm(att_x_video)
            x_video = att_x_video.view(batch_size, length, 8, feature_dim) # [batch, length, class_dim, features] 
            
            # Extract Features
            if self.training:
                self.feature_maps = x_video
            # INTER
            x_video = x_video.permute(0,2,3,1).reshape(batch_size*8, 512, length) # [batch x class_dim, length, features] 
            # x_video = self.temporal['video'](x_video)
            # x_video = self.bn['video'](x_video)
        

            x_video = x_video.transpose(1, 2) # [batch x class_dim, features, length]
            x_video = x_video.view(batch_size,8,length, 512) # [batch x class_dim, features, length]
            x_video = self.fc1(x_video).view(batch_size*8, length, 128).transpose(1, 2) # [batch x class_dim, features, length]
            x_video = self.bn1(x_video).transpose(1, 2).view(batch_size,8,length, 128) # [batch, class_dim, features, length]
            
            x_video = F.leaky_relu(x_video)
            x_video = self.fc2(x_video)
            x_video = x_video.squeeze(3).transpose(1,2) # [batch, length, class_dim]
            return x_video
            
        else:
            if 'video' in X:
                x_video = self.output_layer(x_video)
                # x_video = l2_norm(x_video)
                x_video = x_video.view(batch_size, length, feature_dim).permute(0,2,1)
            
            # Extract features
            if self.training and 'video' in X:
                self.res50_features = x_video
            
            if 'context' in self.args.modality and 'video' in self.args.modality:
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
                x_context = self.context_dropout(x_context)

                x_context = self.fc1(x_context.transpose(1, 2)).transpose(1, 2)
                x_context = self.bn1(x_context).transpose(1, 2)       
                
                x_context = F.leaky_relu(x_context)
                x_context = self.fc2(x_context)
                
                return x_context 
            
            elif 'video' in self.args.modality:
                x_video = self.temporal['video'](x_video)
                x_video = self.bn['video'](x_video)
                
                x_video = self.fc1(x_video.transpose(1, 2)).transpose(1, 2)
                x_video = self.bn1(x_video).transpose(1, 2)       
                
                x_video = F.leaky_relu(x_video)
                x_video = self.fc2(x_video)
                
                return x_video
            
            else:
                raise ValueError('Modality not recognized.')
            
        
    

class InterAUEncoder(nn.Module):
    def __init__(self, num_au=8, au_emb_dim=768, hidden_size=768, num_heads=6, num_layers=2):
        super(InterAUEncoder, self).__init__()
        
        # Layer to project AU embeddings into a higher hidden size if needed
        self.au_embedding_proj = nn.Linear(au_emb_dim, hidden_size)
        
        # Transformer encoder layer: the base block for Inter-AU information exchange
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        
        # Stack multiple layers of the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer to project from hidden size back to AU feature dimension
        self.output_layer = nn.Linear(hidden_size, au_emb_dim)
    
    def forward(self, au_features):
        """
        au_features: Tensor of shape (batch_size, num_au, au_emb_dim)
        """
        
        # Project AU features to hidden size
        au_proj = self.au_embedding_proj(au_features)
        
        # Pass through the transformer encoder (transformer requires sequence-first input)
        au_proj = au_proj.transpose(0, 1)  # Convert to (num_au, batch_size, hidden_size)
        inter_au_output = self.transformer_encoder(au_proj)
        inter_au_output = inter_au_output.transpose(0, 1)  # Back to (batch_size, num_au, hidden_size)
        
        return inter_au_output

class InterSentenceEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, dropout=0.4):
        """
        Args:
            embed_size: Size of the input embeddings (same as from Intra-AU encoder).
            num_layers: Number of Transformer layers in the Inter-AU encoder.
            num_heads: Number of attention heads in the multi-head attention mechanism.
            num_AUs: Number of Action Units (AUs) for embedding exchange.
            dropout: Dropout rate for the encoder.
        """
        super(InterSentenceEncoder, self).__init__()
        
        # Transformer layers for Inter-AU encoding
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_size)
        
    def forward(self, intra_au_embeddings):
        """
        Args:
            intra_au_embeddings: Embeddings from the Intra-AU encoder.
                                 Shape: (NS, batch_size, embed_size)
                                 where NS = number of AUs.
        
        Returns:
            Final output of the Inter-AU encoder.
            Shape: (NS, batch_size, embed_size)
        """
        # Apply transformer layers on the input embeddings
        out = intra_au_embeddings
        for layer in self.transformer_layers:
            out = layer(out)
        
        out = out + intra_au_embeddings
        # Final layer normalization
        out = self.layer_norm(out)
        
        return out
class InterConnexions(nn.Module):

    def __init__(self,
                 device,
                 num_feats_modality: list,
                 num_out_feats,
                 semantic_context_path,
                 args
                 ):
        """ Instantiate feature concatenation fusion instance.

        Args:
            num_feats_modality (list): Number of features per modality.
            num_out_feats (int): Number of output features.
            num_out_feats (int): Number of output features.
        """

        super(InterConnexions, self).__init__()
        self.args = args
        self.device = device
        self.sentence_embeddings = self.create_sentence_embeddings(semantic_context_path)
        self.inter_sentence_encoder = InterAUEncoder(hidden_size=768, num_layers=2, num_heads=6)
        
        self.linear = nn.ModuleDict()
        
        for emotion in range (8):
            emotion_str = str(emotion)
            self.linear[emotion_str] = Sequential(Linear(768, 512),
                # Dropout(0.4),
                # BatchNorm1d(512),
                nn.ReLU())
    
    def create_sentence_embeddings(self, semantic_context_path):
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        model = SentenceTransformer("bert-base-nli-mean-tokens")
        
        sentences = []
        with open(semantic_context_path, 'r') as file:
            contextual_info = json.load(file)
        for emotion in contextual_info['emotions']: 
            sentences.append(emotion['description'])

        sentences_label =[]
        for emotion in contextual_info['emotions']: 
            sentences_label.append(emotion['name'])
        
        sentence_embeddings = model.encode(sentences)
        
        # embedding_matrix = nn.Embedding(8, 768)
        # embedding_matrix.weight.data.copy_(torch.from_numpy(sentence_embeddings).to(self.device)).to(self.device)

        return sentence_embeddings

    def save_tsne(self, sentence_embeddings, name):
        # PATHS
        path_to_save = os.path.join(self.args.save_path, f'{name}.png')
        
        tsne = TSNE(n_components=2, perplexity=3, random_state=42)
        reduced_embeddings = tsne.fit_transform(sentence_embeddings)

        emotions = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        colors = sns.color_palette("hsv", len(emotions)) 
        
        plt.figure(figsize=(10, 8))
        for i, emotion in enumerate(emotions):
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], 
                        color=colors[i], label=emotion, s=40)

        plt.legend(markerscale=1, loc='best', title='T-SNE of Sentence Embeddings')
        plt.title('t-SNE of Sentence Embeddings')
        plt.savefig(path_to_save)
        plt.clf()
        
    def save_umap(self, sentence_embeddings, name):
        # PATHS
        path_to_save = os.path.join(self.args.save_path, f'{name}.png')
        # UMAP
        
        umap = UMAP(n_components=2,random_state=42)
        reduced_embeddings = umap.fit_transform(sentence_embeddings)

        emotions = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        colors = sns.color_palette("hsv", len(emotions)) 
        
        plt.figure(figsize=(10, 8))
        for i, emotion in enumerate(emotions):
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], 
                        color=colors[i], label=emotion, s=40)

        # Plot max and min
        plt.xlim(reduced_embeddings[:, 0].min() - 1, reduced_embeddings[:, 0].max() + 1)
        plt.ylim(reduced_embeddings[:, 1].min() - 1, reduced_embeddings[:, 1].max() + 1)
        plt.legend(markerscale=1, loc='best', title='T-SNE of Sentence Embeddings')
        plt.title('t-SNE of Sentence Embeddings')
        plt.savefig(path_to_save)
        plt.clf()
        
    def save_pcc(self, sentence_embeddings, name):
        # PATHS
        path_to_save = os.path.join(self.args.save_path, f'{name}.png')
        
        pcc = np.corrcoef(sentence_embeddings)
        sns.heatmap(pcc, annot=True, cmap='coolwarm',vmin=0.5, vmax=1, cbar=True)
        # Have a static heat map 
        # Label each column and row with the corresponding emotion
        emotions = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        plt.xticks(ticks=np.arange(len(emotions)) + 0.5, labels=emotions, rotation=45)
        plt.yticks(ticks=np.arange(len(emotions)) + 0.5, labels=emotions, rotation=0)
        plt.title('PCC of Sentence Embeddings')
        plt.savefig(path_to_save)
        plt.clf()
    
    def forward(self, batch_size):
        """ Forward pass

        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
            labels (list): List of labels (BS x SeqLen x 1).
        """
        
        # If you want to save the t-SNE plot:
        if self.args.save_tsne_pcc_inter_connexions:
            self.save_tsne(self.sentence_embeddings, 'tsne_before_TAM')
            self.save_umap(self.sentence_embeddings, 'umaps_before_TAM')
            self.save_pcc(self.sentence_embeddings, 'pcc_matrix_before_TAM')
        
        sentence_embeddings = torch.tensor(self.sentence_embeddings).unsqueeze(0).expand(batch_size, 8, 768).to(self.device)
        
        if (self.args.compute_tam):
            inter_sentence_embeddings = self.inter_sentence_encoder(sentence_embeddings)
        else: 
            inter_sentence_embeddings = sentence_embeddings
            
        if self.args.save_tsne_pcc_inter_connexions:
            self.save_tsne(inter_sentence_embeddings[0].cpu().detach().numpy(), 'tsne_after_TAM')
            self.save_umap(inter_sentence_embeddings[0].cpu().detach().numpy(), 'umaps_after_TAM')
            self.save_pcc(inter_sentence_embeddings[0].cpu().detach().numpy(), 'pcc_matrix_after_TAM')
        
        inter_sentence_embeddings = inter_sentence_embeddings.permute(1,0,2)
        
        _inter_sentence_embeddings = []
        for emotion in range (8):
            emotion_str = str(emotion)
            _inter_sentence_embeddings.append(self.linear[emotion_str](inter_sentence_embeddings[emotion]))
        
        inter_sentence_embeddings = torch.stack(_inter_sentence_embeddings)
        inter_sentence_embeddings = inter_sentence_embeddings.permute(1,0,2)
        return inter_sentence_embeddings