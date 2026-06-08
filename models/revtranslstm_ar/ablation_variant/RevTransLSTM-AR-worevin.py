import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class CrossAttention(nn.Module):
    """以 FullAttention 包裝的標準 Cross-Attention。"""

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
    Ablation: worevin — 移除 RevIN 的 RevTransLSTM-AR。

    保留 Transformer Encoder、LSTM、Cross-Attention 與 closed-loop latent feedback，
    但 encoder/decoder 輸入不做可逆 instance normalization，輸出也不再反標準化。
    用於評估 RevIN 對分布偏移的修正效果。
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

        # Closed-loop feedback：將 c_out 維的預測投影回 d_model 以餵入下一步 LSTM
        self.out_proj = nn.Linear(configs.c_out, configs.d_model)

    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
    ):
        # 1. Encoder：產生 cross-attention 使用的 memory（無 RevIN 標準化）
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 2. Decoder seed：取 label_len 段 embedding 的最後一步作為 LSTM 起始輸入
        dec_input = x_dec[:, :self.label_len, :]
        dec_mark = x_mark_dec[:, :self.label_len, :]
        dec_embed = self.dec_embedding(dec_input, dec_mark)

        lstm_input = dec_embed[:, -1:, :]                       # [B, 1, D]
        hidden = None
        outputs = []

        # 3. 自迴歸解碼：每步以 closed-loop latent feedback 餵入下一步
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

            # 預測投影回 d_model，與 cross-attention 輸出殘差相加作為下一步輸入
            pred_feedback = self.out_proj(pred)                 # [B, 1, D]
            lstm_input = attn_out + pred_feedback

        outputs = torch.cat(outputs, dim=1)                     # [B, pred_len, c_out]

        return outputs
