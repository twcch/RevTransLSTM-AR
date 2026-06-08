import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Ablation: worevinandarfeedbackandcrossattention — 同時移除 RevIN、AR feedback
    與 Cross-Attention 的 RevTransLSTM-AR。

    僅保留 Transformer Encoder（其輸出不被使用）、LSTM 與 projection；
    解碼為 open-loop，每步以 `attn_out = lstm_out` 直接 projection 出預測。
    作為對照基線，用於評估全部組件同時關閉時的退化程度。
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

        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
    ):
        # 1. Encoder：輸出不參與解碼，但保留以維持 encoder 路徑訓練
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 2. Decoder seed：取 label_len 段 embedding 的最後一步作為 LSTM 起始輸入
        dec_input = x_dec[:, :self.label_len, :]
        dec_mark = x_mark_dec[:, :self.label_len, :]
        dec_embed = self.dec_embedding(dec_input, dec_mark)

        lstm_input = dec_embed[:, -1:, :]                       # [B, 1, D]
        hidden = None
        outputs = []

        # 3. 自迴歸解碼：無 cross-attention、無 feedback
        for i in range(self.pred_len):
            lstm_out, hidden = self.lstm(lstm_input, hidden)    # [B, 1, D]

            attn_out = lstm_out                                 # [B, 1, D]

            pred = self.projection(attn_out)                    # [B, 1, c_out]
            outputs.append(pred)

            # 無 AR feedback：直接以 lstm_out 餵入下一步 LSTM
            lstm_input = attn_out

        outputs = torch.cat(outputs, dim=1)                     # [B, pred_len, c_out]

        return outputs
