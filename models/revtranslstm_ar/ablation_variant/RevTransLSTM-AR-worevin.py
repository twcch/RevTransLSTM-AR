import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


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
    Ablation: worevin — RevTransLSTM-AR with RevIN removed.

    Keeps the Transformer Encoder, the LSTM, Cross-Attention and the closed-loop
    latent feedback, but the encoder/decoder inputs are not reversibly instance-
    normalized and the output is not denormalized. Used to measure how much RevIN
    corrects for distribution shift.
    """

    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in

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

        # Closed-loop feedback: project the c_out-dim prediction back to d_model
        # so it can be added to the next step's LSTM input.
        self.out_proj = nn.Linear(configs.c_out, configs.d_model)

    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
    ):
        # 1. Encoder: produce the cross-attention memory (no RevIN normalization).
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 2. Decoder seed: embed the label_len segment and take its last step as
        #    the LSTM's initial input.
        dec_input = x_dec[:, :self.label_len, :]
        dec_mark = x_mark_dec[:, :self.label_len, :]
        dec_embed = self.dec_embedding(dec_input, dec_mark)

        lstm_input = dec_embed[:, -1:, :]                       # [B, 1, D]
        hidden = None
        outputs = []

        # 3. Autoregressive decoding: each step feeds the next via closed-loop
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

        return outputs
