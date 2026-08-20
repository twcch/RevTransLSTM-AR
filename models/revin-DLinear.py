import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    RevIN + DLinear

    DLinear decomposes each series into a trend component (moving average) and a
    seasonal/residual component, then maps each component from the input length to
    the prediction length with a single linear layer along the time axis, and sums
    the two. RevIN normalizes the input before decomposition and denormalizes the
    forecast afterwards, which helps under distribution shift.

    DLinear paper: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether to use a separate linear layer per variate
            (channel-independent) instead of one shared linear layer.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        # For non-forecasting tasks the output length equals the input length.
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer (moving-average kernel):
        # returns (seasonal = x - moving_avg, trend = moving_avg).
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        # RevIN
        self.revin = RevIN(configs.enc_in)

        if self.individual:
            # One linear layer per channel for each component.
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                # Initialize as a uniform-average map (every output step is the
                # mean of the inputs); the model learns to deviate from this.
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            # Shared linear layer across all channels for each component.
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Same uniform-average initialization as above.
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # x: [B, seq_len, channels]
        seasonal_init, trend_init = self.decompsition(x)
        # Move the time axis last so the linear layers map over time: [B, C, L].
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            # Apply the per-channel linear layers one channel at a time.
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        # Recombine the two components and restore [B, pred_len, C].
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # Flatten the time and channel axes to (batch_size, pred_len * channels);
        # for classification pred_len == seq_len, matching the projection input size.
        output = enc_out.reshape(enc_out.shape[0], -1)
        # Project to class logits: (batch_size, num_classes).
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            x_enc = self.revin(x_enc, 'norm')
            dec_out = self.forecast(x_enc)
            dec_out = self.revin(dec_out, 'denorm')
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            x_enc = self.revin(x_enc, 'norm')
            dec_out = self.imputation(x_enc)
            dec_out = self.revin(dec_out, 'denorm')
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            x_enc = self.revin(x_enc, 'norm')
            dec_out = self.anomaly_detection(x_enc)
            dec_out = self.revin(dec_out, 'denorm')
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            # Normalize only (output is class logits, so no denorm).
            x_enc = self.revin(x_enc, 'norm')
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
