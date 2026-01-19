import torch
import torch.nn as nn

from layers.Embed import DataEmbedding

# =============================================================================
#  Research Component: RevIN (Reversible Instance Normalization)
#  保留此部分以確保數據處理與主模型一致，保證公平比較
# =============================================================================
class RevIN(nn.Module):
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
            self._get_statistics(x)
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
    Baseline Model: Pure LSTM (w/o Transformer Encoder & Cross-Attention)
    
    用於消融實驗對比：證明引入 Transformer Encoder 提供的 Global Context 是有效的。
    此模型僅依賴 LSTM 進行時序推理。
    """
    def __init__(self, configs):
        super().__init__()

        # ---- core lengths ----
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # ---- dims ----
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in
        
        # [RevIN] 保持開啟，確保與主模型在同一起跑線
        use_revin = getattr(configs, 'rev_in', True) 
        self.revin = RevIN(configs.enc_in) if use_revin else None

        # ---- embedding ----
        # 只需要 Decoder Embedding，因為我們不再使用 Encoder 的輸入
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # ---- LSTM Decoder (As Main Predictor) ----
        self.lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=getattr(configs, "d_layers", 1),
            batch_first=True,
            dropout=configs.dropout if getattr(configs, "d_layers", 1) > 1 else 0.0
        )

        # ---- Output head ----
        self.projection = nn.Linear(configs.d_model, configs.c_out)
        
        # ---- Feedback Projection ----
        # 將預測結果投影回 d_model 維度，作為下一步的輸入 (Standard AR)
        self.out_proj = nn.Linear(configs.c_out, configs.d_model)

    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
    ):
        """
        Pure LSTM Forward Pass
        x_enc: 被忽略 (因為沒有 Encoder)
        x_dec: 主要輸入
        """
        
        # 1. [RevIN] Normalization
        if self.revin is not None:
            # 這裡我們只處理 x_dec，因為 x_enc 不會被使用
            x_dec_input_vals = x_dec[:, :, :self.enc_in]
            x_dec_input_vals = self.revin(x_dec_input_vals, 'norm')
            x_dec = torch.cat([x_dec_input_vals, x_dec[:, :, self.enc_in:]], dim=-1)

        # 2. Prepare Decoder Seed (Initial Context)
        # 取出 label_len 長度的已知數據作為啟動序列
        dec_input = x_dec[:, :self.label_len, :]                
        dec_mark = x_mark_dec[:, :self.label_len, :]
        dec_embed = self.dec_embedding(dec_input, dec_mark)     

        # LSTM Start Token (取最後一個時間步)
        lstm_input = dec_embed[:, -1:, :]                       # [B, 1, D]
        hidden = None
        outputs = []

        # 3. Autoregressive Decoding Loop (Pure LSTM)
        for i in range(self.pred_len):
            # A. LSTM Step
            # 這裡沒有 Cross-Attention 的干涉，純粹靠 LSTM 的 hidden state 記憶
            lstm_out, hidden = self.lstm(lstm_input, hidden)    # [B, 1, D]

            # B. Projection (Prediction)
            pred = self.projection(lstm_out)                    # [B, 1, c_out]
            outputs.append(pred)

            # C. [Feedback Logic]
            # 下一步的輸入 = 當前預測值的投影 (自回歸)
            # 在主模型中是 attn_out + pred_feedback，這裡沒了 attn_out，直接用 pred_feedback
            # 也可以加上 lstm_out 做 residual，這裡選擇直接將預測值餵回，符合標準 LSTM 邏輯
            pred_feedback = self.out_proj(pred)                 # [B, 1, D]
            
            # 選項: 加上 Residual Connection 讓訓練更穩定 (模擬主模型結構)
            lstm_input = pred_feedback
            # lstm_input = pred_feedback # 如果想要最純粹的 AR，可以用這一行

        outputs = torch.cat(outputs, dim=1)                     # [B, pred_len, c_out]

        # 4. [RevIN] De-Normalization
        if self.revin is not None:
            outputs = self.revin(outputs, 'denorm')

        return outputs