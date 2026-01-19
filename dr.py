import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 設定區 (請修改這裡) =================
# 1. 設定你的真實值檔案路徑 (隨便選一個資料夾的 true.npy 即可，因為真實值是一樣的)
true_path = 'results/long_term_forecast_2330TW_30_1_TransLSTM_AR_custom_ftMS_sl30_ll7_pl1_dm128_nh8_el1_dl1_df1024_expand2_dc4_fc5_ebtimeF_dtTrue_Exp_0/true.npy'

# 2. 設定你要比較的模型路徑 (可以放多個)
title = "NDX"
model_paths = {
    'RevTransLSTM-AR': 'results/long_term_forecast_NDX_30_1_TransLSTM_AR_custom_ftMS_sl30_ll7_pl1_dm128_nh8_el1_dl1_df1024_expand2_dc4_fc5_ebtimeF_dtTrue_Exp_0/pred.npy',
    'DLinear':     'results/backup_results/long_term_forecast_NDX_96_1_DLinear_custom_ftMS_sl30_ll7_pl1_dm24_nh8_el2_dl1_df24_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/pred.npy',
    'TiDE':     'results/backup_results/long_term_forecast_NDX_96_1_TiDE_custom_ftMS_sl30_ll7_pl1_dm24_nh8_el2_dl2_df24_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/pred.npy',
}


# 3. 參數設定
pred_len = 1         # 你跑的預測長度 (pl=1 或 pl=5)
feature_idx = -1     # 通常最後一欄是 Close Price (收盤價)，如果不是請改 0 或其他
sample_start = 0     # 從第幾個樣本開始畫
sample_end = 100     # 畫多少個樣本 (不要太多，不然看不清楚細節)

# =======================================================

def load_data():
    # 載入真實值
    trues = np.load(true_path)
    
    # 載入預測值
    preds = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            preds[name] = np.load(path)
        else:
            print(f"Warning: 找不到檔案 {path}")
    
    return trues, preds

def plot_one_step_ahead(trues, preds):
    """
    畫出「單步預測」的連續曲線 (適合看整體趨勢和滯後)
    雖然是 pl=5，但我們只取每一步的第1個預測點串起來，看誰反應快。
    """
    plt.figure(figsize=(15, 6))
    
    # 取出真實值 (單變量)
    # trues shape: [Samples, PredLen, Features] -> 取每個 window 的第 0 步
    gt = trues[sample_start:sample_end, 0, feature_idx]
    plt.plot(gt, label='Ground Truth', color='black', linewidth=2)

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (name, pred_data) in enumerate(preds.items()):
        # 取出預測值
        pd = pred_data[sample_start:sample_end, 0, feature_idx] # 取第 0 步預測 (t+1)
        
        plt.plot(pd, label=name, color=colors[i % len(colors)], linestyle='--')

    plt.title(f'1 Steps Ahead Prediction Comparison ({title})', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.savefig('comparison_plot_onestep.png')
    plt.show()

def plot_multi_step_trajectory(trues, preds, specific_index):
    """
    畫出「特定時間點的未來 5 步預測」 (適合看轉折點)
    這會畫出一個 Ground Truth 線，然後從某一點射出幾條預測線。
    """
    plt.figure(figsize=(10, 6))
    
    # 畫背景真實線 (前後多畫一點 context)
    context = 10
    gt = trues[specific_index-context : specific_index+pred_len+context, 0, feature_idx]
    x_axis = range(specific_index-context, specific_index+pred_len+context)
    plt.plot(x_axis, gt, label='Ground Truth', color='black', marker='o', markersize=4)

    # 畫預測線 (從 specific_index 開始的未來 pl 步)
    pred_x_axis = range(specific_index, specific_index + pred_len)
    
    colors = ['blue', 'red']
    for i, (name, pred_data) in enumerate(preds.items()):
        # 取出特定樣本的完整 pl 步預測
        pd = pred_data[specific_index, :, feature_idx]
        plt.plot(pred_x_axis, pd, label=name + ' Forecast', color=colors[i], marker='x', linestyle='--', linewidth=2)

    plt.title(f'Multi-Step Forecasting at Index {specific_index}', fontsize=14)
    plt.axvline(x=specific_index, color='gray', linestyle=':', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'comparison_plot_multistep_{specific_index}.png')
    plt.show()

# 執行
trues, preds = load_data()

# 1. 畫整體圖 (找 Lag)
plot_one_step_ahead(trues, preds)

# 2. 畫特寫圖 (自己改 index 找一個轉折點！)
# 你需要試幾個 index，找到一個「股價急跌或急漲」的地方
# 例如：假設第 50 個樣本是轉折點
plot_multi_step_trajectory(trues, preds, specific_index=50)