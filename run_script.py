import time
import os
import sys
from datetime import datetime
import json
import importlib
import argparse
import torch
import random
import numpy as np

# ==================== 配置區域 ====================

# 可用模型列表 (根據 exp/exp_basic.py 中的 model_dict)
AVAILABLE_MODELS = [
    "TimesNet",
    "Autoformer",
    "Transformer",
    "Nonstationary_Transformer",
    "DLinear",
    "FEDformer",
    "Informer",
    "LightTS",
    "Reformer",
    "ETSformer",
    "PatchTST",
    "Pyraformer",
    "MICN",
    "Crossformer",
    "FiLM",
    "iTransformer",
    "Koopa",
    "TiDE",
    "FreTS",
    "MambaSimple",
    "TimeMixer",
    "TSMixer",
    "SegRNN",
    "TimeXer",
    "PAttn",
    "TransLSTM_AR",
    "TransLSTM_AR_wo1",
    "TransLSTM_AR_wo2",
    "TransLSTM_AR_wo3",
    "TransLSTM_AR_wo4",
]

# 可用資料集配置
DATASET_CONFIGS = {
    "ETTh1": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    "ETTh2": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh2.csv",
        "data": "ETTh2",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    "ETTm1": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm1.csv",
        "data": "ETTm1",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    "ETTm2": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm2.csv",
        "data": "ETTm2",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    "electricity": {
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "data": "custom",
        "enc_in": 321,
        "dec_in": 321,
        "c_out": 321,
    },
    "traffic": {
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "data": "custom",
        "enc_in": 862,
        "dec_in": 862,
        "c_out": 862,
    },
    "weather": {
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "data": "custom",
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
    },
    "exchange_rate": {
        "root_path": "./dataset/exchange_rate/",
        "data_path": "exchange_rate.csv",
        "data": "custom",
        "enc_in": 8,
        "dec_in": 8,
        "c_out": 8,
    },
    "2330TW": {
        "root_path": "./data/",
        "data_path": "2330TW.csv",
        "data": "custom",
        "enc_in": 4,
        "dec_in": 4,
        "c_out": 4,
    },
    "AAPL": {
        "root_path": "./data/",
        "data_path": "AAPL.csv",
        "data": "custom",
        "enc_in": 4,
        "dec_in": 4,
        "c_out": 4,
    },
    "SPX": {
        "root_path": "./data/",
        "data_path": "SPX.csv",
        "data": "custom",
        "enc_in": 4,
        "dec_in": 4,
        "c_out": 4,
    },
    "SOX": {
        "root_path": "./data/",
        "data_path": "SOX.csv",
        "data": "custom",
        "enc_in": 4,
        "dec_in": 4,
        "c_out": 4,
    },
    "NDX": {
        "root_path": "./data/",
        "data_path": "NDX.csv",
        "data": "custom",
        "enc_in": 4,
        "dec_in": 4,
        "c_out": 4,
    },
}

# 任務類型配置
TASK_CONFIGS = {
    "long_term_forecast": {
        "task_name": "long_term_forecast",
        "features": "MS",
        "seq_len": 30,
        "label_len": 7,
        "pred_lens": [1, 5],
    },
    "short_term_forecast": {
        "task_name": "short_term_forecast",
        "features": "M",
        "seq_len": 96,
        "label_len": 48,
        "pred_lens": [96],
    },
    "imputation": {
        "task_name": "imputation",
        "features": "M",
        "seq_len": 96,
        "label_len": 0,
        "pred_lens": [0],
    },
    "anomaly_detection": {
        "task_name": "anomaly_detection",
        "features": "M",
        "seq_len": 100,
        "label_len": 0,
        "pred_lens": [0],
    },
    "classification": {
        "task_name": "classification",
        "features": "M",
        "seq_len": 96,
        "label_len": 0,
        "pred_lens": [0],
    },
}

# 模型特定配置
MODEL_CONFIGS = {
    "Transformer": {"e_layers": 1, "d_layers": 1, "factor": 2},
    "Informer": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "Autoformer": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "FEDformer": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "PatchTST": {"e_layers": 3, "d_layers": 1, "factor": 3, "n_heads": 4},
    "TimesNet": {"e_layers": 2, "d_layers": 1, "d_model": 32, "d_ff": 32, "top_k": 5},
    "DLinear": {"e_layers": 2, "d_layers": 1},
    "LightTS": {"e_layers": 2, "d_layers": 1},
    "ETSformer": {"e_layers": 2, "d_layers": 2},
    "FiLM": {"e_layers": 2, "d_layers": 1},
    "Crossformer": {"e_layers": 2, "d_layers": 1},
    "Pyraformer": {"e_layers": 2, "d_layers": 1},
    "MICN": {"e_layers": 2, "d_layers": 1},
    "Koopa": {"e_layers": 2, "d_layers": 1},
    "TimeXer": {"e_layers": 1, "d_model": 256, "d_ff": 512},
    "SegRNN": {"seg_len": 48, "d_model": 512, "dropout": 0.5},
    "TiDE": {"e_layers": 2, "d_layers": 2, "d_model": 256, "d_ff": 256},
    "TimeMixer": {"e_layers": 2,"d_model": 16,"d_ff": 32,"down_sampling_layers": 3,"down_sampling_window": 2,"down_sampling_method": "avg",},
    "MambaSimple": {"e_layers": 2, "d_model": 128, "d_ff": 16, "d_conv": 4, "expand": 2,},
    "iTransformer": {"e_layers": 3, "d_model": 512, "d_ff": 512},
    "PAttn": {"n_heads": 4},
    "Reformer": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "Nonstationary_Transformer": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "FreTS": {"e_layers": 2, "d_layers": 1},
    "TSMixer": {"e_layers": 2, "d_layers": 1},
    "TransLSTM_AR": {"d_ff": 1024, "d_model": 128, "e_layers": 1, "d_layers": 1, "factor": 5, "dropout": 0.05},
    "TransLSTM_AR_wo1": {"d_ff": 512, "d_model": 256, "e_layers": 2, "d_layers": 1, "factor": 5, "dropout": 0.1},
    "TransLSTM_AR_wo2": {"d_ff": 512, "d_model": 256, "e_layers": 2, "d_layers": 1, "factor": 5, "dropout": 0.1},
    "TransLSTM_AR_wo3": {"d_ff": 2048, "d_model": 512, "e_layers": 2, "d_layers": 8, "factor": 5, "dropout": 0.6},
    "TransLSTM_AR_wo4": {"d_ff": 2048, "d_model": 512, "e_layers": 2, "d_layers": 8, "factor": 5, "dropout": 0.6},
}


def get_default_args():
    """獲取與 run.py 相同的預設參數"""
    args = argparse.Namespace(
        # basic config
        task_name="long_term_forecast",
        is_training=1,
        model_id="test",
        model="Transformer",
        # data loader
        data="ETTh1",
        root_path="./dataset/ETT-small/",
        data_path="ETTh1.csv",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        # forecasting task
        seq_len=96,
        label_len=48,
        pred_len=96,
        seasonal_patterns="Monthly",
        inverse=False,
        # imputation task
        mask_rate=0.25,
        # anomaly detection task
        anomaly_ratio=0.25,
        # model define
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=48,
        # optimization
        num_workers=10,
        itr=1,
        train_epochs=10,
        batch_size=32,
        patience=3,
        learning_rate=0.0001,
        des="Exp",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        # GPU
        use_gpu=True,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0,1,2,3",
        # de-stationary projector params
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        # metrics
        use_dtw=False,
        # Augmentation
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        # TimeXer
        patch_len=16,
        # PAttn
        pos=1,
    )
    return args


class ExperimentRunner:
    def __init__(
        self,
        models: list,
        datasets: list,
        task: str = "long_term_forecast",
        pred_lens: list = None,
        gpu_id: int = 0,
        is_training: bool = True,
        itr: int = 1,
        batch_size: int = 32,
        train_epochs: int = 10,
        learning_rate: float = 0.0001,
        patience: int = 3,
        des: str = "Exp",
        output_dir: str = "./experiment_logs",
        custom_args: dict = None,
    ):
        """
        初始化實驗運行器

        Args:
            models: 模型列表 e.g., ['Transformer', 'PatchTST']
            datasets: 資料集列表 e.g., ['ETTh1', 'ETTm1']
            task: 任務類型
            pred_lens: 預測長度列表，若為None則使用預設
            gpu_id: GPU ID
            is_training: 是否訓練
            itr: 實驗迭代次數
            batch_size: batch大小
            train_epochs: 訓練輪數
            learning_rate: 學習率
            patience: 早停耐心值
            des: 實驗描述
            output_dir: 日誌輸出目錄
            custom_args: 自定義參數字典
        """
        self.models = models
        self.datasets = datasets
        self.task = task
        self.pred_lens = pred_lens or TASK_CONFIGS.get(task, {}).get("pred_lens", [96])
        self.gpu_id = gpu_id
        self.is_training = is_training
        self.itr = itr
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.des = des
        self.output_dir = output_dir
        self.custom_args = custom_args or {}

        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)

        # 實驗記錄
        self.experiment_logs = []

        # 設置隨機種子
        self._set_seed(2021)

        # 導入實驗類
        self._import_exp_classes()

    def _set_seed(self, seed):
        """設置隨機種子"""
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _import_exp_classes(self):
        """動態導入實驗類"""
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        from exp.exp_imputation import Exp_Imputation
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        from exp.exp_classification import Exp_Classification

        self.exp_classes = {
            "long_term_forecast": Exp_Long_Term_Forecast,
            "short_term_forecast": Exp_Short_Term_Forecast,
            "imputation": Exp_Imputation,
            "anomaly_detection": Exp_Anomaly_Detection,
            "classification": Exp_Classification,
        }

    def build_args(self, model: str, dataset: str, pred_len: int) -> argparse.Namespace:
        """構建實驗參數"""
        args = get_default_args()

        dataset_config = DATASET_CONFIGS.get(dataset, {})
        task_config = TASK_CONFIGS.get(self.task, {})
        model_config = MODEL_CONFIGS.get(model, {})

        # 基本參數
        args.task_name = task_config.get("task_name", self.task)
        args.is_training = int(self.is_training)
        args.model = model
        args.model_id = f'{dataset}_{task_config.get("seq_len", 30)}_{pred_len}'

        # 資料集參數
        args.root_path = dataset_config.get("root_path", "./")
        args.data_path = dataset_config.get("data_path", "")
        args.data = dataset_config.get("data", "custom")
        args.features = task_config.get("features", "MS")

        # 序列參數
        args.seq_len = task_config.get("seq_len", 30)
        args.label_len = task_config.get("label_len", 7)
        args.pred_len = pred_len

        # 模型結構參數
        args.enc_in = dataset_config.get("enc_in", 4)
        args.dec_in = dataset_config.get("dec_in", 4)
        args.c_out = dataset_config.get("c_out", 4)

        # 模型特定參數
        for key, value in model_config.items():
            setattr(args, key, value)

        # 訓練參數
        args.batch_size = self.batch_size
        args.train_epochs = self.train_epochs
        args.learning_rate = self.learning_rate
        args.patience = self.patience
        args.itr = self.itr
        args.des = self.des

        # GPU設置
        args.gpu = self.gpu_id
        args.use_gpu = torch.cuda.is_available()

        # 自定義參數
        for key, value in self.custom_args.items():
            setattr(args, key, value)

        # 設置設備
        if torch.cuda.is_available() and args.use_gpu:
            args.device = torch.device(f"cuda:{args.gpu}")
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.device = torch.device("mps")
            else:
                args.device = torch.device("cpu")

        return args

    def _get_setting(self, args, ii):
        """生成實驗設定字串"""
        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )
        return setting

    def run_single_experiment(self, model: str, dataset: str, pred_len: int) -> dict:
        """運行單個實驗"""

        print("\n" + "=" * 80)
        print(f"🚀 Running: Model={model}, Dataset={dataset}, PredLen={pred_len}")
        print("=" * 80 + "\n")

        args = self.build_args(model, dataset, pred_len)

        # 獲取對應的實驗類
        Exp = self.exp_classes.get(self.task)
        if Exp is None:
            raise ValueError(f"Unknown task: {self.task}")

        start_time = time.time()
        start_datetime = datetime.now()
        status = "success"
        error_msg = ""

        try:
            if args.is_training:
                for ii in range(args.itr):
                    setting = self._get_setting(args, ii)

                    exp = Exp(args)

                    print(
                        f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>"
                    )
                    exp.train(setting)

                    print(
                        f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                    )
                    exp.test(setting)

                    # 清理GPU緩存
                    if args.gpu_type == "mps":
                        torch.backends.mps.empty_cache()
                    elif args.gpu_type == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                ii = 0
                setting = self._get_setting(args, ii)

                exp = Exp(args)

                print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.test(setting, test=1)

                if args.gpu_type == "mps":
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            status = "failed"
            error_msg = str(e)
            import traceback

            traceback.print_exc()

        end_time = time.time()
        end_datetime = datetime.now()
        duration = end_time - start_time

        log_entry = {
            "model": model,
            "dataset": dataset,
            "pred_len": pred_len,
            "task": self.task,
            "status": status,
            "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "duration_formatted": self.format_duration(duration),
            "error": error_msg,
        }

        self.experiment_logs.append(log_entry)

        # 打印結果摘要
        status_emoji = "✅" if status == "success" else "❌"
        print(f"\n{status_emoji} Completed: {model} on {dataset} (pred_len={pred_len})")
        print(f'⏱️  Duration: {log_entry["duration_formatted"]}')
        print(f"📊 Status: {status}")

        if status != "success":
            print(f"⚠️  Error: {error_msg[:500]}")

        return log_entry

    def run_all(self) -> list:
        """運行所有實驗組合"""

        total_experiments = len(self.models) * len(self.datasets) * len(self.pred_lens)
        current_exp = 0

        total_start_time = time.time()

        print("\n" + "#" * 80)
        print("#" + " " * 30 + "EXPERIMENT START" + " " * 32 + "#")
        print("#" * 80)
        print(f"\n📋 Total Experiments: {total_experiments}")
        print(f"🔧 Models: {self.models}")
        print(f"📁 Datasets: {self.datasets}")
        print(f"📏 Prediction Lengths: {self.pred_lens}")
        print(f"🎯 Task: {self.task}")
        print(f"🖥️  GPU: {self.gpu_id}")
        print("\n")

        for model in self.models:
            for dataset in self.datasets:
                for pred_len in self.pred_lens:
                    current_exp += 1
                    print(f"\n📌 Progress: [{current_exp}/{total_experiments}]")

                    self.run_single_experiment(model, dataset, pred_len)

                    # 每次實驗後保存日誌
                    self.save_logs()

        total_duration = time.time() - total_start_time

        # 打印總結
        self.print_summary(total_duration)

        return self.experiment_logs

    def format_duration(self, seconds: float) -> str:
        """格式化時間"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def save_logs(self):
        """保存實驗日誌"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON格式
        json_path = os.path.join(self.output_dir, f"experiment_log_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.experiment_logs, f, indent=2, ensure_ascii=False)

        # 保存CSV格式的摘要
        csv_path = os.path.join(self.output_dir, f"experiment_summary_{timestamp}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(
                "model,dataset,pred_len,task,status,duration_seconds,duration_formatted,start_time,end_time,error\n"
            )
            for log in self.experiment_logs:
                error_escaped = (
                    log.get("error", "").replace(",", ";").replace("\n", " ")[:200]
                )
                f.write(
                    f'{log["model"]},{log["dataset"]},{log["pred_len"]},{log["task"]},'
                    f'{log["status"]},{log["duration_seconds"]},{log["duration_formatted"]},'
                    f'{log["start_time"]},{log["end_time"]},{error_escaped}\n'
                )

        print(f"\n💾 Logs saved to: {json_path}")

    def print_summary(self, total_duration: float):
        """打印實驗總結"""

        print("\n" + "#" * 80)
        print("#" + " " * 30 + "EXPERIMENT SUMMARY" + " " * 30 + "#")
        print("#" * 80)

        success_count = sum(
            1 for log in self.experiment_logs if log["status"] == "success"
        )
        failed_count = sum(
            1 for log in self.experiment_logs if log["status"] != "success"
        )
        total_count = len(self.experiment_logs)

        print(f"\n📊 Results:")
        print(f"   ✅ Success: {success_count}/{total_count}")
        print(f"   ❌ Failed: {failed_count}/{total_count}")
        print(f"\n⏱️  Total Time: {self.format_duration(total_duration)}")

        if self.experiment_logs:
            avg_duration = sum(
                log["duration_seconds"] for log in self.experiment_logs
            ) / len(self.experiment_logs)
            print(f"📈 Average per experiment: {self.format_duration(avg_duration)}")

        # 按模型統計
        print("\n📋 Results by Model:")
        model_stats = {}
        for log in self.experiment_logs:
            model = log["model"]
            if model not in model_stats:
                model_stats[model] = {"success": 0, "failed": 0, "total_time": 0}
            model_stats[model]["total_time"] += log["duration_seconds"]
            if log["status"] == "success":
                model_stats[model]["success"] += 1
            else:
                model_stats[model]["failed"] += 1

        for model, stats in model_stats.items():
            status_emoji = "✅" if stats["failed"] == 0 else "⚠️"
            print(
                f'   {status_emoji} {model}: {stats["success"]} success, {stats["failed"]} failed, '
                f'time: {self.format_duration(stats["total_time"])}'
            )

        # 顯示失敗的實驗
        if failed_count > 0:
            print("\n❌ Failed Experiments:")
            for log in self.experiment_logs:
                if log["status"] != "success":
                    print(
                        f'   - {log["model"]} on {log["dataset"]} (pred_len={log["pred_len"]}): {log["error"][:100]}'
                    )

        print("\n" + "#" * 80 + "\n")


def list_available():
    """列出可用的模型和資料集"""
    print("\n📋 Available Models:")
    for model in AVAILABLE_MODELS:
        config = MODEL_CONFIGS.get(model, {})
        print(f"   - {model}: {config}")

    print("\n📋 Available Datasets:")
    for dataset, config in DATASET_CONFIGS.items():
        print(f'   - {dataset}: enc_in={config["enc_in"]}')

    print("\n📋 Available Tasks:")
    for task, config in TASK_CONFIGS.items():
        print(f'   - {task}: pred_lens={config.get("pred_lens", "N/A")}')


# ==================== 直接運行區域 ====================

if __name__ == "__main__":
    
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device name:", torch.cuda.get_device_name(0))

    # ===== 配置你的實驗 =====

    # 選擇模型
    # "DLinear", "Informer", "Autoformer", "PatchTST", "Transformer", "iTransformer", "TiDE", "MambaSimple", "PAttn"
    MODELS = ["TransLSTM_AR_wo4", "TransLSTM_AR", "TransLSTM_AR_wo3", "TransLSTM_AR_wo2", "TransLSTM_AR_wo1"]

    # 選擇資料集
    DATASETS = ["2330TW", "AAPL", "SPX", "SOX", "NDX"]

    # 選擇任務類型: 'long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection', 'classification'
    TASK = "long_term_forecast"

    # 預測長度 (設為 None 使用任務預設值)
    PRED_LENS = [1, 5]  # 或設為 None

    # GPU ID
    GPU_ID = 0

    # 訓練參數
    IS_TRAINING = True
    BATCH_SIZE = 32
    TRAIN_EPOCHS = 50
    LEARNING_RATE = 0.0001
    PATIENCE = 5
    ITR = 1

    # 實驗描述
    DES = "Exp"

    # 日誌輸出目錄
    OUTPUT_DIR = "./experiment_logs"

    # 自定義參數 (可選)
    CUSTOM_ARGS = {
        "gpu_type": "cuda",
        "seq_len": 30,
        "label_len": 7,
        "target": "close",
    }

    # ===== 運行實驗 =====

    # 取消下面的註解來列出可用選項
    # list_available()

    # 創建實驗運行器
    runner = ExperimentRunner(
        models=MODELS,
        datasets=DATASETS,
        task=TASK,
        pred_lens=PRED_LENS,
        gpu_id=GPU_ID,
        is_training=IS_TRAINING,
        itr=ITR,
        batch_size=BATCH_SIZE,
        train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        des=DES,
        output_dir=OUTPUT_DIR,
        custom_args=CUSTOM_ARGS,
    )

    # 運行所有實驗
    runner.run_all()
