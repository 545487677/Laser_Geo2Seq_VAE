# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F


class GRUDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        batch_first: bool = True,
        decoder_layers: int = 6,
        **kwargs
    ) -> None:
        super(GRUDecoder, self).__init__()
        self.decoder = nn.GRU(latent_dim // 2, latent_dim // 2, decoder_layers, batch_first=batch_first)

    def forward(self, x=None):
        out, _ = self.decoder(x)
        return out


class MLPDecoder(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
        input_dim=3, 
        hidden_dim=256, 
        num_layers=3,
        **kwargs,
        ):
        super(MLPDecoder, self).__init__()
        
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 3))  # Final layer to output 3 coordinates
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    