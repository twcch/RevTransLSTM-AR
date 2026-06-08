import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN).

    Normalizes / denormalizes each instance with statistics detached from the
    autograd graph. The ``'transform'`` mode reuses the statistics cached by
    ``'norm'`` so the decoder input is normalized consistently with the encoder
    input.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # Learnable per-feature scale and shift applied after standardization.
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            # Compute and cache statistics, then standardize (called once per forward).
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'transform':
            # Reuse the statistics already cached by 'norm'; do not recompute or overwrite.
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_statistics(self):
        self.mean = None
        self.stdev = None

    def _get_statistics(self, x):
        # Reduce over every dimension except batch (0) and feature (last);
        # for a [B, L, F] tensor this reduces the time axis, giving [B, 1, F].
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.affine_weight.detach())
        x = x * self.stdev
        x = x + self.mean
        return x


class CrossAttention(nn.Module):
    """Standard (non-causal) cross-attention wrapped around ``FullAttention``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        self.attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=False
            ),
            d_model=d_model,
            n_heads=n_heads
        )

    def forward(self, query, key, value, attn_mask=None):
        out, _ = self.attention(query, key, value, attn_mask=attn_mask)
        return out


class Model(nn.Module):
    """
    Ablation: woarfeedback — RevTransLSTM-AR with the AR feedback removed.

    Keeps RevIN, the Transformer Encoder, the LSTM and Cross-Attention, but during
    decoding each step feeds ``attn_out`` directly into the next LSTM step
    (open-loop) instead of folding the prediction back into the latent space.
    Used to measure the contribution of the closed-loop latent feedback to AR
    decoding.
    """

    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in

        # Enable RevIN by default when configs does not provide `rev_in`.
        use_revin = getattr(configs, 'rev_in', True)
        self.revin = RevIN(configs.enc_in) if use_revin else None

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=getattr(configs, "d_layers", 1),
            batch_first=True,
            dropout=configs.dropout if getattr(configs, "d_layers", 1) > 1 else 0.0
        )

        self.cross_attention = CrossAttention(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
            factor=configs.factor
        )

        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
    ):
        # 1. RevIN normalization: x_enc computes and caches the statistics,
        #    x_dec reuses that same cached set of statistics.
        if self.revin is not None:
            x_enc = self.revin(x_enc, 'norm')

            # Normalize only the first enc_in features of dec_in (the real values);
            # keep the remaining columns (e.g. time marks) untouched and concat back.
            x_dec_input_vals = x_dec[:, :, :self.enc_in]
            x_dec_input_vals = self.revin(x_dec_input_vals, 'transform')
            x_dec = torch.cat([x_dec_input_vals, x_dec[:, :, self.enc_in:]], dim=-1)

        # 2. Encoder: produce the memory consumed by cross-attention.
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 3. Decoder seed: embed the label_len segment and take its last step as
        #    the LSTM's initial input.
        dec_input = x_dec[:, :self.label_len, :]
        dec_mark = x_mark_dec[:, :self.label_len, :]
        dec_embed = self.dec_embedding(dec_input, dec_mark)

        lstm_input = dec_embed[:, -1:, :]                       # [B, 1, D]
        hidden = None
        outputs = []

        # 4. Autoregressive decoding: open-loop — the next input is the
        #    cross-attention output directly.
        for i in range(self.pred_len):
            lstm_out, hidden = self.lstm(lstm_input, hidden)    # [B, 1, D]

            attn_out = self.cross_attention(
                query=lstm_out,
                key=enc_out,
                value=enc_out,
                attn_mask=dec_enc_mask
            )                                                   # [B, 1, D]

            pred = self.projection(attn_out)                    # [B, 1, c_out]
            outputs.append(pred)

            # No AR feedback: feed attn_out straight into the next LSTM step.
            lstm_input = attn_out

        outputs = torch.cat(outputs, dim=1)                     # [B, pred_len, c_out]

        # 5. RevIN denormalization: map predictions back to the original scale.
        if self.revin is not None:
            outputs = self.revin(outputs, 'denorm')

        return outputs
