import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class RevIN(nn.Module):
    """Reversible Instance Normalization：以 detach 後的統計量做標準化／反標準化。"""

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            # 計算並儲存統計量後做標準化（每個 forward 流程僅呼叫一次）
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'transform':
            # 沿用 'norm' 已算好的統計量做標準化，不重算、不覆寫
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


class Model(nn.Module):
    """
    Ablation: wocrossattention — 移除 Cross-Attention 的 RevTransLSTM-AR。

    保留 RevIN、Transformer Encoder、LSTM 與 closed-loop latent feedback，但解碼時
    不查詢 encoder memory，直接以 `lstm_out` 作為 context（`attn_out = lstm_out`）。
    Encoder 仍會被計算（影響 encoder embedding 路徑訓練），但其輸出不參與解碼。
    用於評估 cross-attention 對 AR 解碼的貢獻。
    """

    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in

        # 若 configs 未提供 rev_in，預設啟用 RevIN
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

        self.projection = nn.Linear(configs.d_model, configs.c_out)

        # Closed-loop feedback：將 c_out 維的預測投影回 d_model 以餵入下一步 LSTM
        self.out_proj = nn.Linear(configs.c_out, configs.d_model)

    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
    ):
        # 1. RevIN 標準化：x_enc 計算並儲存統計量，x_dec 沿用同一份統計量
        if self.revin is not None:
            x_enc = self.revin(x_enc, 'norm')

            # 僅對 dec_in 前 enc_in 個特徵做標準化，其餘欄位保留原值後拼回
            x_dec_input_vals = x_dec[:, :, :self.enc_in]
            x_dec_input_vals = self.revin(x_dec_input_vals, 'transform')
            x_dec = torch.cat([x_dec_input_vals, x_dec[:, :, self.enc_in:]], dim=-1)

        # 2. Encoder：輸出不參與解碼，但保留以維持 encoder 路徑訓練
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 3. Decoder seed：取 label_len 段 embedding 的最後一步作為 LSTM 起始輸入
        dec_input = x_dec[:, :self.label_len, :]
        dec_mark = x_mark_dec[:, :self.label_len, :]
        dec_embed = self.dec_embedding(dec_input, dec_mark)

        lstm_input = dec_embed[:, -1:, :]                       # [B, 1, D]
        hidden = None
        outputs = []

        # 4. 自迴歸解碼：無 cross-attention，context 直接取 LSTM 輸出
        for i in range(self.pred_len):
            lstm_out, hidden = self.lstm(lstm_input, hidden)    # [B, 1, D]

            attn_out = lstm_out                                 # [B, 1, D]

            pred = self.projection(attn_out)                    # [B, 1, c_out]
            outputs.append(pred)

            # 預測投影回 d_model，與 context 殘差相加作為下一步輸入
            pred_feedback = self.out_proj(pred)                 # [B, 1, D]
            lstm_input = attn_out + pred_feedback

        outputs = torch.cat(outputs, dim=1)                     # [B, pred_len, c_out]

        # 5. RevIN 反標準化：還原至原始尺度
        if self.revin is not None:
            outputs = self.revin(outputs, 'denorm')

        return outputs
