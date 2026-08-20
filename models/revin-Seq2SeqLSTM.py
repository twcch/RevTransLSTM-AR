import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    RevIN + Seq2Seq-LSTM

    Encoder: multi-layer LSTM compresses the (RevIN-normalized) input
             sequence into the final (h, c) state.
    Decoder: multi-layer LSTM initialized from the encoder state, decoding
             autoregressively for ``pred_len`` steps (recursive multi-step),
             seeded with the last observed time step.
    RevIN:   normalize before the encoder, denormalize after the decoder.
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

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout if configs.e_layers > 1 else 0.0,
            batch_first=True,
        )

        # Decoder LSTM (input is the previous step's value, dim = enc_in)
        self.decoder_lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout if configs.e_layers > 1 else 0.0,
            batch_first=True,
        )
        self.projection = nn.Linear(configs.d_model, configs.enc_in)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.cls_projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def encode(self, x):
        # x: [B, seq_len, enc_in]
        enc_out, (h, c) = self.encoder_lstm(x)
        return enc_out, (h, c)

    def forecast(self, x_enc):
        _, (h, c) = self.encode(x_enc)

        # Seed the decoder with the last observed step
        dec_input = x_enc[:, -1:, :]  # [B, 1, enc_in]
        outputs = []
        for _ in range(self.pred_len):
            dec_out, (h, c) = self.decoder_lstm(dec_input, (h, c))
            step = self.projection(dec_out)  # [B, 1, enc_in]
            outputs.append(step)
            dec_input = step  # recursive multi-step

        return torch.cat(outputs, dim=1)  # [B, pred_len, enc_in]

    def imputation(self, x_enc):
        enc_out, _ = self.encode(x_enc)
        return self.projection(enc_out)  # [B, seq_len, enc_in]

    def anomaly_detection(self, x_enc):
        enc_out, _ = self.encode(x_enc)
        return self.projection(enc_out)  # [B, seq_len, enc_in]

    def classification(self, x_enc):
        enc_out, _ = self.encode(x_enc)
        output = self.act(enc_out)
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        # (batch_size, num_classes)
        output = self.cls_projection(output)
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
