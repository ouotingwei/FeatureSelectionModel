from typing import List
import sys
import numpy as np
import yaml
import cv2 as cv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/home/wei/deep_feature_selection/code/training/extractors/orbslam2_features/lib')
from orbslam2_features import ORBextractor


def MLP(channels: List[int], do_bn: bool = False) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Encoding of geometric properties using MLP """
    def __init__(self, keypoint_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP([keypoint_dim] + layers + [feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, kpts):
        if self.use_dropout:
            return self.dropout(self.encoder(kpts))
        return self.encoder(kpts)


class DescriptorEncoder(nn.Module):
    """ Encoding of visual descriptor using MLP """
    def __init__(self, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP([feature_dim] + layers + [feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, descs):
        residual = descs
        if self.use_dropout:
            return residual + self.dropout(self.encoder(descs))
        return residual + self.encoder(descs)


class AFTAttention(nn.Module):
    """ Attention-free attention """
    def __init__(self, d_model: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.dim = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = torch.sigmoid(q)
        k = k.T
        k = torch.softmax(k, dim=-1)
        k = k.T
        kv = (k * v).sum(dim=-2, keepdim=True)
        x = q * kv
        x = self.proj(x)
        if self.use_dropout:
            x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.mlp = MLP([feature_dim, feature_dim*2, feature_dim])
        self.layer_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        if self.use_dropout:
            x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class AttentionalLayer(nn.Module):
    def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1):
        super().__init__()
        self.attn = AFTAttention(feature_dim, dropout=dropout, p=p)
        self.ffn = PositionwiseFeedForward(feature_dim, dropout=dropout, p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class AttentionalNN(nn.Module):
    def __init__(self, feature_dim: int, layer_num: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalLayer(feature_dim, dropout=dropout, p=p)
            for _ in range(layer_num)])

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            desc = layer(desc)
        return desc


class FeatureBooster(nn.Module):
    default_config = {
        'descriptor_dim': 128,
        'keypoint_encoder': [32, 64, 128],
        'Attentional_layers': 3,
        'last_activation': 'relu',
        'l2_normalization': True,
        'output_dim': 128
    }

    def __init__(self, config, dropout=False, p=0.1, use_kenc=True, use_cross=True):
        super().__init__()

        self.config = {**self.default_config, **config}
        self.use_kenc = use_kenc
        self.use_cross = use_cross

        if use_kenc:
            self.kenc = KeypointEncoder(
                self.config['keypoint_dim'], self.config['descriptor_dim'], self.config['keypoint_encoder'], dropout=dropout)

        if self.config.get('descriptor_encoder', False):
            self.denc = DescriptorEncoder(
                self.config['descriptor_dim'], self.config['descriptor_encoder'], dropout=dropout)
        else:
            self.denc = None

        if self.use_cross:
            self.attn_proj = AttentionalNN(
                feature_dim=self.config['descriptor_dim'], layer_num=self.config['Attentional_layers'], dropout=dropout)

        self.final_proj = nn.Linear(
            self.config['descriptor_dim'], self.config['output_dim'])

        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

        self.layer_norm = nn.LayerNorm(self.config['descriptor_dim'], eps=1e-6)

        if self.config.get('last_activation', False):
            if self.config['last_activation'].lower() == 'relu':
                self.last_activation = nn.ReLU()
            elif self.config['last_activation'].lower() == 'sigmoid':
                self.last_activation = nn.Sigmoid()
            elif self.config['last_activation'].lower() == 'tanh':
                self.last_activation = nn.Tanh()
            else:
                raise Exception('Not supported activation "%s".' % self.config['last_activation'])
        else:
            self.last_activation = None

    def forward(self, desc, kpts):
        ## Self boosting
        # Descriptor MLP encoder
        if self.denc is not None:
            desc = self.denc(desc)
        # Geometric MLP encoder
        if self.use_kenc:
            desc = desc + self.kenc(kpts)

            if self.use_dropout:
                desc = self.dropout(desc)
        
        ## Cross boosting
        # Multi-layer Transformer network.
        if self.use_cross:
            desc = self.attn_proj(self.layer_norm(desc))


        ## Post processing
        # Final MLP projection
        desc = self.final_proj(desc)
        if self.last_activation is not None:
            desc = self.last_activation(desc)

        # L2 normalization
        if self.config['l2_normalization']:
            desc = F.normalize(desc, dim=-1)

        return desc
    

def normalize_keypoints(keypoints, image_shape):
    x0 = image_shape[1] / 2
    y0 = image_shape[0] / 2
    scale = max(image_shape) * 0.7
    kps = np.array(keypoints)
    kps[:, 0] = (keypoints[:, 0] - x0) / scale
    kps[:, 1] = (keypoints[:, 1] - y0) / scale
    return kps 

def booster_process(image):
    # set CUDA
    use_cuda = torch.cuda.is_available()

    # set torch grad ( speed up ! )
    torch.set_grad_enabled(False)
    
    start_time = time.time()

    # bgr -> gray
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # orb extractor
    feature_extractor = ORBextractor(1000, 1.2, 8)

    # set FeatureBooster
    config_file = '/home/wei/deep_feature_selection/code/training/config.yaml'
    with open(str(config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Model
    feature_booster = FeatureBooster(config['ORB+Boost-B'])
    if use_cuda:
        feature_booster.cuda()

    feature_booster.eval()

    feature_booster.load_state_dict(torch.load('/home/wei/deep_feature_selection/code/training/ORB+Boost-B.pth'))

    kps_tuples, descriptors = feature_extractor.detectAndCompute(image)
    
    # convert keypoints 
    keypoints = [cv.KeyPoint(*kp) for kp in kps_tuples]
    keypoints = np.array(
        [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints], 
        dtype=np.float32
    )

    # boosted the descriptor using trained model
    kps = normalize_keypoints(keypoints, image.shape)
    kps = torch.from_numpy(kps.astype(np.float32))
    descriptors = np.unpackbits(descriptors, axis=1, bitorder='little')
    descriptors = descriptors * 2.0 - 1.0
    descriptors = torch.from_numpy(descriptors.astype(np.float32))

    if use_cuda:
        kps = kps.cuda()
        descriptors = descriptors.cuda()

    out = feature_booster(descriptors, kps)
    out = (out >= 0).cpu().detach().numpy()
    descriptors = np.packbits(out, axis=1, bitorder='little')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("feature_booster: ", elapsed_time, "sec")

    return keypoints, descriptors

def convert_to_cv_keypoints(keypoints):
    cv_keypoints = []
    for kp in keypoints:
        x, y = kp[0], kp[1]
        size = kp[2]
        angle = kp[3]
        cv_kp = cv.KeyPoint(x, y, size, angle)
        cv_keypoints.append(cv_kp)
    return cv_keypoints