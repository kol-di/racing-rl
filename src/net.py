import torch.nn as nn
import torch
import numpy as np


class A2CNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(), 
            nn.Flatten(start_dim=1)
        )

        conv_out_size = self._get_conv_out(input_dim)
        self.policy_mu = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions), 
            nn.Tanh()
        )

        self.policy_var = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions), 
            nn.Softplus()
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv_embedding(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv_embedding(x)
        return self.policy_mu(conv_out), self.policy_var(conv_out) + 1e-6, self.value(conv_out)
    
    def forward_policy(self, x):
        conv_out = self.conv_embedding(x)
        return self.policy_mu(conv_out), self.policy_var(conv_out) + 1e-6