import numpy as np
import matplotlib.pyplot as plt
import os
from utils.metrics import metric

# ================= 設定區 (請修改這裡) =================
# 1. 設定你的真實值檔案路徑
true_path = 'results/long_term_forecast_2330TW_30_1_TransLSTM_AR_custom_ftMS_sl30_ll7_pl1_dm128_nh8_el1_dl1_df1024_expand2_dc4_fc5_ebtimeF_dtTrue_Exp_0/true.npy'

# 2. 設定你要比較的模型路徑
model_paths = {
    'RevTransLSTM-AR': 'results/long_term_forecast_2330TW_30_1_TransLSTM_AR_custom_ftMS_sl30_ll7_pl1_dm128_nh8_el1_dl1_df1024_expand2_dc4_fc5_ebtimeF_dtTrue_Exp_0/pred.npy',
    'Transformer':     'results/backup_results/long_term_forecast_2330TW_96_1_Transformer_custom_ftMS_sl30_ll7_pl1_dm256_nh8_el1_dl1_df512_expand2_dc4_fc2_ebtimeF_dtTrue_Exp_0/pred.npy',
    'Informer':     'results/backup_results/long_term_forecast_2330TW_96_1_Informer_custom_ftMS_sl30_ll7_pl1_dm24_nh8_el2_dl1_df24_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/pred.npy',
    'Autoformer':     'results/backup_results/long_term_forecast_2330TW_96_1_Autoformer_custom_ftMS_sl30_ll7_pl1_dm24_nh8_el2_dl1_df24_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/pred.npy',
}

# 3. 參數設定
pred_len = 1
feature_idx = -1
sample_start = 0
sample_end = 100

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

def calculate_metrics(trues, preds):
    """
    計算所有模型的評估指標
    """
    metrics_results = {}
    
    for name, pred_data in preds.items():
        # 使用 metric 函數計算指標
        mae, mse, rmse, mape, mspe, r2 = metric(pred_data, trues)
        
        metrics_results[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'MSPE': mspe,
            'R2': r2
        }
        
        # 印出結果
        print(f"\n{name}:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAPE: {mape:.6f}")
        print(f"  MSPE: {mspe:.6f}")
        print(f"  R2:   {r2:.6f}")
    
    return metrics_results

def plot_metrics_comparison(metrics_results):
    """
    繪製各模型的指標比較圖
    """
    models = list(metrics_results.keys())
    metric_names = ['MSE','R2']
    
    # 創建 2x3 的子圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx // 3, idx % 3]
        
        # 收集每個模型的該指標值
        values = [metrics_results[model][metric_name] for model in models]
        
        # 繪製長條圖
        bars = ax.bar(models, values, color=['blue', 'red', 'green', 'orange', 'purple'][:len(models)])
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_heatmap(metrics_results):
    """
    繪製指標熱力圖
    """
    models = list(metrics_results.keys())
    metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'R2']
    
    # 創建數據矩陣
    data = np.zeros((len(models), len(metric_names)))
    for i, model in enumerate(models):
        for j, metric_name in enumerate(metric_names):
            data[i, j] = metrics_results[model][metric_name]
    
    # 正規化數據 (每個指標除以該指標的最大值)
    data_norm = data / data.max(axis=0)
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data_norm, cmap='RdYlGn_r', aspect='auto')
    
    # 設定軸標籤
    plt.xticks(range(len(metric_names)), metric_names)
    plt.yticks(range(len(models)), models)
    
    # 添加顏色條
    plt.colorbar(im, label='Normalized Value')
    
    # 添加數值標籤
    for i in range(len(models)):
        for j in range(len(metric_names)):
            text = plt.text(j, i, f'{data[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.title('Model Metrics Heatmap (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_one_step_ahead(trues, preds):
    """
    畫出「單步預測」的連續曲線
    """
    plt.figure(figsize=(15, 6))
    
    gt = trues[sample_start:sample_end, 0, feature_idx]
    plt.plot(gt, label='Ground Truth', color='black', linewidth=2)

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (name, pred_data) in enumerate(preds.items()):
        pd = pred_data[sample_start:sample_end, 0, feature_idx]
        plt.plot(pd, label=name, color=colors[i % len(colors)], linestyle='--')

    plt.title(f'One-Step Ahead Prediction Comparison (Sample {sample_start}-{sample_end})', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.savefig('comparison_plot_onestep.png')
    plt.show()

def plot_multi_step_trajectory(trues, preds, specific_index):
    """
    畫出「特定時間點的未來預測」
    """
    plt.figure(figsize=(10, 6))
    
    context = 10
    gt = trues[specific_index-context : specific_index+pred_len+context, 0, feature_idx]
    x_axis = range(specific_index-context, specific_index+pred_len+context)
    plt.plot(x_axis, gt, label='Ground Truth', color='black', marker='o', markersize=4)

    pred_x_axis = range(specific_index, specific_index + pred_len)
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, (name, pred_data) in enumerate(preds.items()):
        pd = pred_data[specific_index, :, feature_idx]
        plt.plot(pred_x_axis, pd, label=name + ' Forecast', color=colors[i % len(colors)], marker='x', linestyle='--', linewidth=2)

    plt.title(f'Multi-Step Forecasting at Index {specific_index}', fontsize=14)
    plt.axvline(x=specific_index, color='gray', linestyle=':', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'comparison_plot_multistep_{specific_index}.png')
    plt.show()

# 執行
trues, preds = load_data()

# 1. 計算並顯示指標
metrics_results = calculate_metrics(trues, preds)

# 2. 繪製指標比較圖
plot_metrics_comparison(metrics_results)

# 3. 繪製指標熱力圖
plot_metrics_heatmap(metrics_results)

# 4. 畫整體圖
plot_one_step_ahead(trues, preds)

# 5. 畫特寫圖
plot_multi_step_trajectory(trues, preds, specific_index=50)