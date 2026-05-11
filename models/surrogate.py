"""U-Net Surrogate Model for Instant Flow Field Prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, residual: bool = True):
        super().__init__()
        self.residual = residual
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        
        if residual and in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x) if self.residual else None
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + identity
        return self.act(out)


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    Focuses on relevant features from encoder.
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetSurrogate(nn.Module):
    """
    U-Net based surrogate model for instant flow field prediction.
    
    Input: Condition channels (e.g., obstacle mask, Reynolds number, BCs)
    Output: Flow fields (u, v, p)
    """
    
    def __init__(
        self,
        in_channels: int = 4,   # (obstacle_mask, Re, bc_u, bc_v)
        out_channels: int = 3,  # (u, v, p)
        features: List[int] = None,
        use_attention: bool = True,
    ):
        super().__init__()
        
        if features is None:
            features = [32, 64, 128, 256]
        
        self.use_attention = use_attention
        n_levels = len(features)
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for feat in features:
            self.encoders.append(ConvBlock(in_ch, feat))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = feat
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        in_ch = features[-1] * 2
        for feat in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, feat, kernel_size=2, stride=2)
            )
            if use_attention:
                self.attention_gates.append(AttentionGate(feat, feat, feat // 2))
            self.decoders.append(ConvBlock(feat * 2, feat))
            in_ch = feat
        
        # Output projection
        self.output = nn.Sequential(
            nn.Conv2d(features[0], features[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features[0], out_channels, 1),
        )
        
        # Parameter embedding network
        self.param_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, in_channels, H, W) condition maps
            params: (batch, n_params) physical parameters (Re, nu, etc.)
        
        Returns:
            (batch, out_channels, H, W) predicted flow fields
        """
        # Encoder path
        skip_connections = []
        
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            
            skip = skip_connections[i]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Attention gating
            if self.use_attention and i < len(self.attention_gates):
                skip = self.attention_gates[i](x, skip)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.output(x)
    
    def predict(
        self,
        obstacle_mask: np.ndarray,
        re: float = 100.0,
        bc_u: float = 1.0,
        bc_v: float = 0.0,
        device: str = 'cpu'
    ) -> Dict[str, np.ndarray]:
        """
        Quick prediction from condition inputs.
        
        Args:
            obstacle_mask: (H, W) boolean obstacle mask
            re: Reynolds number
            bc_u, bc_v: Boundary velocity components
        
        Returns:
            Dict with 'u', 'v', 'p' fields
        """
        self.eval()
        
        H, W = obstacle_mask.shape
        
        # Build input channels
        ch0 = obstacle_mask.astype(np.float32)
        ch1 = np.full((H, W), re / 1000.0, dtype=np.float32)  # Normalized Re
        ch2 = np.full((H, W), bc_u, dtype=np.float32)
        ch3 = np.full((H, W), bc_v, dtype=np.float32)
        
        x = np.stack([ch0, ch1, ch2, ch3], axis=0)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.forward(x_tensor).cpu().numpy()[0]
        
        return {
            'u': output[0],
            'v': output[1],
            'p': output[2],
        }


class ConditionalUNet(UNetSurrogate):
    """
    Conditional U-Net with FiLM (Feature-wise Linear Modulation).
    
    Physical parameters modulate feature maps at each encoder level:
    γ, β = MLP(params)
    h = γ * h + β  (per-channel scaling and shifting)
    
    This allows the network to adapt to different Reynolds numbers,
    viscosities, etc. without separate models.
    """
    
    def __init__(self, n_params: int = 4, **kwargs):
        super().__init__(**kwargs)
        
        features = kwargs.get('features', [32, 64, 128, 256])
        
        # FiLM generators for each encoder level
        self.film_generators = nn.ModuleList()
        for feat in features:
            self.film_generators.append(
                nn.Sequential(
                    nn.Linear(n_params, 64),
                    nn.ReLU(),
                    nn.Linear(64, feat * 2),  # γ and β
                )
            )
    
    def forward(
        self, x: torch.Tensor, params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if params is None:
            return super().forward(x)
        
        skip_connections = []
        
        for i, (encoder, pool) in enumerate(zip(self.encoders, self.pools)):
            x = encoder(x)
            
            # Apply FiLM conditioning
            if params is not None:
                film = self.film_generators[i](params)
                gamma = film[:, :x.shape[1]].unsqueeze(-1).unsqueeze(-1)
                beta = film[:, x.shape[1]:].unsqueeze(-1).unsqueeze(-1)
                x = gamma * x + beta
            
            skip_connections.append(x)
            x = pool(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            if self.use_attention and i < len(self.attention_gates):
                skip = self.attention_gates[i](x, skip)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.output(x)
