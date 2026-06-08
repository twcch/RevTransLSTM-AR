import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    RevIN + GRU

    Encoder: multi-layer GRU over the (RevIN-normalized) input sequence.
    Forecast: project the last hidden state to the full prediction horizon.
    RevIN:    normalize before the GRU, denormalize after the projection.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.num_layers = configs.e_layers

        # RevIN
        self.revin = RevIN(configs.enc_in)

        # GRU encoder
        self.gru = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout if configs.e_layers > 1 else 0.0,
            batch_first=True,
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # map last hidden state -> [pred_len, enc_in]
            self.projection = nn.Linear(
                configs.d_model, self.pred_len * configs.enc_in)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # per-step reconstruction
            self.projection = nn.Linear(configs.d_model, configs.enc_in)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # x: [B, seq_len, enc_in]
        gru_out, _ = self.gru(x)  # [B, seq_len, d_model]
        return gru_out

    def forecast(self, x_enc):
        # Direct multi-step: project the last hidden state to the whole horizon at once.
        gru_out = self.encoder(x_enc)
        last_step = gru_out[:, -1, :]  # [B, d_model]
        out = self.projection(last_step)  # [B, pred_len * enc_in]
        out = out.reshape(out.size(0), self.pred_len, self.enc_in)
        return out

    def imputation(self, x_enc):
        gru_out = self.encoder(x_enc)
        return self.projection(gru_out)  # [B, seq_len, enc_in]

    def anomaly_detection(self, x_enc):
        gru_out = self.encoder(x_enc)
        return self.projection(gru_out)  # [B, seq_len, enc_in]

    def classification(self, x_enc):
        gru_out = self.encoder(x_enc)
        output = self.act(gru_out)
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        # (batch_size, num_classes)
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
            x_enc = self.revin(x_enc, 'norm')
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
