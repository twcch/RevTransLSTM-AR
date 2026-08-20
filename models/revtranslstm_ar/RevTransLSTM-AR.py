import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN).

    Normalizes / denormalizes each instance with statistics that are detached
    from the autograd graph, so the per-instance mean and standard deviation do
    not receive gradients. This is the standard remedy for distribution shift in
    non-stationary time series (Kim et al., ICLR 2022).

    This in-model copy adds a ``'transform'`` mode (not present in the shared
    ``layers/RevIN``) so the decoder input can be normalized with the statistics
    that were already computed and cached from the encoder input.
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
            # Invert the affine transform first. The denominator adds eps times a
            # detached copy of affine_weight: this keeps the division well-conditioned
            # while preventing gradients from flowing back through the stability term.
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.affine_weight.detach())
        x = x * self.stdev
        x = x + self.mean
        return x


class CrossAttention(nn.Module):
    """Standard (non-causal) cross-attention wrapped around ``FullAttention``.

    The query comes from the decoder LSTM state while keys/values come from the
    encoder memory, so each decoding step can attend over the whole input window.
    ``mask_flag=False`` means no causal mask is applied.
    """

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
        # AttentionLayer returns (output, attn_weights); we keep only the output.
        out, _ = self.attention(query, key, value, attn_mask=attn_mask)
        return out


class Model(nn.Module):
    """
    RevTransLSTM-AR

    Architecture:
    1. RevIN: reversible instance normalization on the encoder/decoder inputs.
    2. Transformer Encoder: extracts a global representation of the input
       sequence that serves as the cross-attention memory.
    3. LSTM + Cross-Attention: decodes autoregressively, one step at a time.
    4. Closed-loop latent feedback: each step's prediction is projected back to
       the latent space by the learnable ``out_proj`` and added (residually) to
       the cross-attention output to form the next step's LSTM input.
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

        # Decoder recurrence: input and hidden size are both d_model so the
        # cross-attention output can be fed straight back in (see feedback below).
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

        # Maps the latent decoder state to the c_out target variables per step.
        self.projection = nn.Linear(configs.d_model, configs.c_out)

        # Closed-loop feedback: project the c_out-dim prediction back to d_model
        # so it can be added to the next step's LSTM input.
        self.out_proj = nn.Linear(configs.c_out, configs.d_model)

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

        # 4. Autoregressive decoding: each step feeds the next via closed-loop
        #    latent feedback.
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

            # Project the prediction back to d_model and add it to the cross-attention
            # output; this closed-loop latent feedback becomes the next step's LSTM input.
            pred_feedback = self.out_proj(pred)                 # [B, 1, D]
            lstm_input = attn_out + pred_feedback

        outputs = torch.cat(outputs, dim=1)                     # [B, pred_len, c_out]

        # 5. RevIN denormalization: map predictions back to the original scale.
        if self.revin is not None:
            outputs = self.revin(outputs, 'denorm')

        return outputs
