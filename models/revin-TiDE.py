import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    """Residual MLP block: two-layer MLP plus a linear skip, then LayerNorm.

    out = LayerNorm( Dropout(fc2(ReLU(fc1(x)))) + fc3(x) )

    ``fc3`` projects the input to ``output_dim`` so the residual add works even
    when ``input_dim != output_dim``. This is TiDE's only building block.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)  # residual projection
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)

    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)  # residual connection
        out = self.ln(out)
        return out


# TiDE
class Model(nn.Module):
    """
    RevIN + TiDE (Time-series Dense Encoder)

    TiDE is an MLP-only encoder-decoder. Per channel it concatenates the lookback
    window with the (encoded) covariates over the whole horizon, runs a stack of
    dense ResBlocks (the encoder), decodes to ``pred_len`` steps, then refines
    each step together with its future covariates (the temporal decoder) and adds
    a linear residual mapping straight from the lookback. RevIN normalizes the
    input and denormalizes the forecast around the whole network.

    TiDE paper: https://arxiv.org/pdf/2304.08424.pdf
    """

    def __init__(self, configs, bias=True, feature_encode_dim=2):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  # L: lookback length
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len  # H: forecast horizon
        self.hidden_dim = configs.d_model
        self.res_hidden = configs.d_model
        self.encoder_num = configs.e_layers
        self.decoder_num = configs.d_layers
        self.freq = configs.freq
        self.feature_encode_dim = feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporalDecoderHidden = configs.d_ff
        self.revin = RevIN(configs.enc_in)
        dropout = configs.dropout

        # Number of raw time-feature channels implied by the sampling frequency.
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

        self.feature_dim = freq_map[self.freq]

        # Encoder input width: the lookback values (seq_len) plus the encoded
        # covariates over the full horizon (seq_len + pred_len steps), flattened.
        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        # Projects per-step covariates (feature_dim) down to feature_encode_dim.
        self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias)
        # Dense encoder: first ResBlock maps flatten_dim -> hidden_dim, then
        # (encoder_num - 1) ResBlocks keep the hidden width.
        self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias),*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.encoder_num-1)))
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Decoder: (decoder_num - 1) hidden ResBlocks then one that expands to
            # decode_dim * pred_len, later reshaped to [B, pred_len, decode_dim].
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, dropout, bias))
            # Temporal decoder: per future step, fuse decoded value + future
            # covariates down to a single scalar.
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            # Global linear residual mapping the lookback directly to the horizon.
            self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        if self.task_name == 'imputation':
            # Same heads as forecasting but the output length is seq_len.
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias))
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)
        if self.task_name == 'anomaly_detection':
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias))
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)

    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # batch_y_mark: covariates over the full window [B, seq_len + pred_len, feature_dim].
        feature = self.feature_encoder(batch_y_mark)
        # Encode lookback values + flattened covariates -> hidden representation.
        hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.pred_len, self.decode_dim)
        # Fuse decoded values with the future covariates, then add the lookback residual.
        dec_out = self.temporalDecoder(torch.cat([feature[:,self.seq_len:], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask):
        # Covariates span only the observed window here (length seq_len).
        feature = self.feature_encoder(x_mark_enc)
        hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.seq_len, self.decode_dim)
        dec_out = self.temporalDecoder(torch.cat([feature[:,:self.seq_len], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask=None):
        '''x_mark_enc is the exogenous dynamic feature described in the original paper'''
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            x_enc = self.revin(x_enc, 'norm')
            # Build the full-horizon covariate sequence: observed marks followed
            # by the future marks (or zeros when none are provided).
            if batch_y_mark is None:
                batch_y_mark = torch.zeros((x_enc.shape[0], self.seq_len+self.pred_len, self.feature_dim)).to(x_enc.device).detach()
            else:
                batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]],dim=1)
            # TiDE is channel-independent: forecast each variate separately and stack.
            dec_out = torch.stack([self.forecast(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark) for feature in range(x_enc.shape[-1])],dim=-1)
            dec_out = self.revin(dec_out, 'denorm')
            return dec_out # [B, H, D]  (H = pred_len)
        if self.task_name == 'imputation':
            x_enc = self.revin(x_enc, 'norm')
            dec_out = torch.stack([self.imputation(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark, mask) for feature in range(x_enc.shape[-1])],dim=-1)
            dec_out = self.revin(dec_out, 'denorm')
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError("Task anomaly_detection for Tide is temporarily not supported")
        if self.task_name == 'classification':
            raise NotImplementedError("Task classification for Tide is temporarily not supported")
        return None
