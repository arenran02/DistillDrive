
import torch
import torch.nn as nn


class DistributionModule(nn.Module):
    """
        A convolutional net that parametrises a diagonal Gaussian distribution.
    """
    def __init__(
        self, in_channels, latent_dim, min_log_sigma=-5, max_log_sigma=5):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        # Encoder
        self.encoder = DistributionEncoder1DV2(
            in_channels,
            self.compress_dim,
        )
        # convolution layer
        self.last_conv = nn.Sequential(
            nn.Conv1d(self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, s_t):
        """
            s_t: [B, N1, D]
        """
        # Learning for every Agent Distriubtion
        encoding = self.encoder(s_t.permute(0, 2, 1)) # [B, D2, N1]
        mu_log_sigma = self.last_conv(encoding).permute(0, 2, 1) #[B, N1, 2* LD]
        mu = mu_log_sigma[:, :, :self.latent_dim]  #[B, N1, LD]
        log_sigma = mu_log_sigma[:, :, self.latent_dim:] #[B, N1, LD]

        # clip the log_sigma value for numerical stability
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma


class DistributionEncoder1DV2(nn.Module):
    """
        Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels * 2, out_channels=in_channels * 2, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels * 2, out_channels=out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s_t):
        """
            s_t: [B, D, N1]
        """
        s_t = self.relu(self.conv1(s_t)) # [B, D, N1] -> [B, D1, N1]
        s_t = self.relu(self.conv2(s_t)) # [B, D1, N1] -> [B, D1, N1]
        s_t = self.conv3(s_t) # [B, D1, N1] -> [B, D2, N1]

        return s_t


class PredictModel(nn.Module):
    """
        predict future states with rnn.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, temporal_frames):
        super().__init__()
        self.temporal_frames = temporal_frames
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels*2)
        self.linear2 = nn.Linear(hidden_channels*2, hidden_channels*4)
        self.linear3 = nn.Linear(hidden_channels*4, out_channels)

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x , h):
        '''
            x: [T, B * N1, LD]
            h: [L, B * N1, LD']
        '''
        # GRU fusion
        x, h = self.gru(x, h)
        # Linear projection
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
