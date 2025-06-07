#!/usr/bin/env python3
"""
実験進捗トラッカー
Experiment Progress Tracker

リアルタイムで実験の進捗・ETA・速度を表示
"""

import time
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

@dataclass
class ProgressStats:
    """進捗統計情報"""
    total_samples: int
    completed_samples: int
    current_dataset: str
    current_lambda: float
    
    start_time: float
    current_time: float
    
    samples_per_second: float
    eta_seconds: float
    eta_formatted: str
    
    progress_percent: float
    elapsed_formatted: str
    
    # 詳細統計
    dataset_progress: Dict[str, Dict]
    lambda_progress: Dict[float, Dict]
    
    # リソース使用量
    avg_processing_time: float
    last_n_samples_time: float

class ExperimentProgressTracker:
    """
    実験進捗トラッカー
    
    リアルタイムで以下を表示：
    - 進捗率 (45.3% 完了)
    - ETA (あと2時間15分)
    - 処理速度 (3.2 samples/sec)
    - データセット別進捗
    - Lambda値別進捗
    """
    
    def __init__(self, total_samples: int, datasets: List[str], lambda_values: List[float]):
        self.total_samples = total_samples
        self.datasets = datasets
        self.lambda_values = lambda_values
        
        # 進捗追跡
        self.completed_samples = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # サンプル処理時間の記録
        self.sample_times: List[float] = []
        self.recent_sample_times: List[float] = []  # 直近N個のサンプル
        self.max_recent_samples = 50  # 直近50サンプルで速度計算
        
        # データセット・Lambda別進捗
        self.dataset_stats = {dataset: {"completed": 0, "total": 0} for dataset in datasets}
        self.lambda_stats = {lam: {"completed": 0, "total": 0} for lam in lambda_values}
        
        # 現在の実行中項目
        self.current_dataset = ""
        self.current_lambda = 0.0
        
        # 表示設定
        self.update_interval = 5.0  # 5秒ごとに更新
        self.last_display_time = 0
        
        print("🚀 実験進捗トラッキング開始")
        print(f"📊 総サンプル数: {total_samples:,}")
        print(f"📋 データセット: {datasets}")
        print(f"🎯 Lambda値: {lambda_values}")
        print("-" * 70)
    
    def start_dataset(self, dataset: str, expected_samples: int):
        """データセット開始"""
        self.current_dataset = dataset
        self.dataset_stats[dataset]["total"] = expected_samples
        print(f"\n📊 {dataset} 開始 ({expected_samples:,} サンプル予定)")
    
    def start_lambda(self, lambda_value: float, expected_samples: int):
        """Lambda値設定開始"""
        self.current_lambda = lambda_value
        self.lambda_stats[lambda_value]["total"] += expected_samples
        print(f"  🎯 λ={lambda_value}: ", end="", flush=True)
    
    def update_progress(self, sample_processing_time: float = None):
        """進捗更新"""
        current_time = time.time()
        
        # サンプル完了
        self.completed_samples += 1
        
        # 処理時間記録
        if sample_processing_time is not None:
            self.sample_times.append(sample_processing_time)
            self.recent_sample_times.append(sample_processing_time)
            
            # 直近N個のサンプルのみ保持
            if len(self.recent_sample_times) > self.max_recent_samples:
                self.recent_sample_times.pop(0)
        
        # データセット・Lambda統計更新
        if self.current_dataset:
            self.dataset_stats[self.current_dataset]["completed"] += 1
        if self.current_lambda:
            self.lambda_stats[self.current_lambda]["completed"] += 1
        
        # 定期的な詳細表示
        if current_time - self.last_display_time >= self.update_interval:
            self._display_detailed_progress()
            self.last_display_time = current_time
        else:
            # 簡易進捗表示
            self._display_simple_progress()
    
    def _display_simple_progress(self):
        """簡易進捗表示"""
        progress_percent = (self.completed_samples / self.total_samples) * 100
        
        # プログレスバー作成
        bar_length = 30
        filled_length = int(bar_length * progress_percent / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        print(f"\r  {bar} {progress_percent:5.1f}% ({self.completed_samples:,}/{self.total_samples:,})", end="", flush=True)
    
    def _display_detailed_progress(self):
        """詳細進捗表示"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 進捗率計算
        progress_percent = (self.completed_samples / self.total_samples) * 100
        
        # 処理速度計算
        if self.completed_samples > 0:
            overall_rate = self.completed_samples / elapsed_time
            
            # 直近の処理速度（より正確）
            if len(self.recent_sample_times) > 1:
                recent_avg_time = sum(self.recent_sample_times) / len(self.recent_sample_times)
                recent_rate = 1.0 / recent_avg_time if recent_avg_time > 0 else overall_rate
            else:
                recent_rate = overall_rate
        else:
            overall_rate = recent_rate = 0
        
        # ETA計算
        remaining_samples = self.total_samples - self.completed_samples
        if recent_rate > 0:
            eta_seconds = remaining_samples / recent_rate
            eta_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
            eta_formatted = eta_time.strftime("%H:%M:%S")
            eta_duration = self._format_duration(eta_seconds)
        else:
            eta_formatted = "計算中..."
            eta_duration = "計算中..."
        
        # 経過時間
        elapsed_formatted = self._format_duration(elapsed_time)
        
        print(f"\n\n📈 【進捗レポート】 {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"┌─ 全体進捗: {progress_percent:5.1f}% ({self.completed_samples:,}/{self.total_samples:,})")
        print(f"├─ 経過時間: {elapsed_formatted}")
        print(f"├─ 残り時間: {eta_duration}")
        print(f"├─ 完了予定: {eta_formatted}")
        print(f"├─ 処理速度: {recent_rate:.2f} samples/sec (直近), {overall_rate:.2f} samples/sec (全体)")
        print(f"└─ 平均処理時間: {sum(self.recent_sample_times)/len(self.recent_sample_times):.2f}秒/sample" if self.recent_sample_times else "")
        
        # データセット別進捗
        print(f"\n📊 データセット別進捗:")
        for dataset, stats in self.dataset_stats.items():
            if stats["total"] > 0:
                ds_progress = (stats["completed"] / stats["total"]) * 100
                status = "🔄" if dataset == self.current_dataset else ("✅" if ds_progress >= 100 else "⏳")
                print(f"  {status} {dataset:12}: {ds_progress:5.1f}% ({stats['completed']:,}/{stats['total']:,})")
        
        # Lambda値別進捗
        print(f"\n🎯 Lambda値別進捗:")
        for lambda_val, stats in self.lambda_stats.items():
            if stats["total"] > 0:
                lam_progress = (stats["completed"] / stats["total"]) * 100
                status = "🔄" if abs(lambda_val - self.current_lambda) < 0.001 else ("✅" if lam_progress >= 100 else "⏳")
                print(f"  {status} λ={lambda_val:4.1f}     : {lam_progress:5.1f}% ({stats['completed']:,}/{stats['total']:,})")
        
        print("-" * 70)
    
    def _format_duration(self, seconds: float) -> str:
        """秒数を読みやすい形式に変換"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}分"
        elif seconds < 86400:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}時間{minutes:.0f}分"
        else:
            days = seconds / 86400
            hours = (seconds % 86400) / 3600
            return f"{days:.0f}日{hours:.0f}時間"
    
    def finish_lambda(self):
        """Lambda値完了"""
        lambda_stats = self.lambda_stats[self.current_lambda]
        print(f" ✅ 完了 ({lambda_stats['completed']:,}サンプル)")
    
    def finish_dataset(self):
        """データセット完了"""
        dataset_stats = self.dataset_stats[self.current_dataset]
        print(f"✅ {self.current_dataset} 完了 ({dataset_stats['completed']:,}サンプル)")
    
    def get_current_stats(self) -> ProgressStats:
        """現在の統計情報取得"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 処理速度計算
        if self.completed_samples > 0:
            overall_rate = self.completed_samples / elapsed_time
            if len(self.recent_sample_times) > 1:
                recent_avg_time = sum(self.recent_sample_times) / len(self.recent_sample_times)
                recent_rate = 1.0 / recent_avg_time if recent_avg_time > 0 else overall_rate
            else:
                recent_rate = overall_rate
        else:
            overall_rate = recent_rate = 0
        
        # ETA計算
        remaining_samples = self.total_samples - self.completed_samples
        if recent_rate > 0:
            eta_seconds = remaining_samples / recent_rate
            eta_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
            eta_formatted = eta_time.strftime("%H:%M:%S")
        else:
            eta_seconds = 0
            eta_formatted = "計算中..."
        
        return ProgressStats(
            total_samples=self.total_samples,
            completed_samples=self.completed_samples,
            current_dataset=self.current_dataset,
            current_lambda=self.current_lambda,
            start_time=self.start_time,
            current_time=current_time,
            samples_per_second=recent_rate,
            eta_seconds=eta_seconds,
            eta_formatted=eta_formatted,
            progress_percent=(self.completed_samples / self.total_samples) * 100,
            elapsed_formatted=self._format_duration(elapsed_time),
            dataset_progress=self.dataset_stats.copy(),
            lambda_progress=self.lambda_stats.copy(),
            avg_processing_time=sum(self.sample_times) / len(self.sample_times) if self.sample_times else 0,
            last_n_samples_time=sum(self.recent_sample_times) / len(self.recent_sample_times) if self.recent_sample_times else 0
        )
    
    def finish_experiment(self):
        """実験完了"""
        total_time = time.time() - self.start_time
        avg_rate = self.completed_samples / total_time if total_time > 0 else 0
        
        print(f"\n" + "="*70)
        print(f"🎉 実験完了!")
        print(f"📊 総サンプル数: {self.completed_samples:,}/{self.total_samples:,}")
        print(f"⏱️  総実行時間: {self._format_duration(total_time)}")
        print(f"🚀 平均処理速度: {avg_rate:.2f} samples/sec")
        if self.sample_times:
            print(f"📈 平均処理時間: {sum(self.sample_times)/len(self.sample_times):.2f}秒/sample")
        print(f"✅ 完了時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

# 使用例とテスト
if __name__ == "__main__":
    import random
    
    # 実験設定
    datasets = ["MMLU", "GSM8K", "HumanEval", "TruthfulQA"]
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    dataset_samples = {"MMLU": 100, "GSM8K": 50, "HumanEval": 20, "TruthfulQA": 30}
    
    total_samples = sum(dataset_samples.values()) * len(lambda_values)
    
    # 進捗トラッカー初期化
    tracker = ExperimentProgressTracker(total_samples, datasets, lambda_values)
    
    print("🧪 進捗トラッカーのテスト実行")
    
    # 実験シミュレーション
    for dataset in datasets:
        num_samples = dataset_samples[dataset]
        tracker.start_dataset(dataset, num_samples * len(lambda_values))
        
        for lambda_val in lambda_values:
            tracker.start_lambda(lambda_val, num_samples)
            
            for sample_idx in range(num_samples):
                # サンプル処理時間をシミュレート
                processing_time = random.uniform(0.5, 3.0)  # 0.5-3.0秒
                time.sleep(0.1)  # 実際の処理のシミュレート
                
                tracker.update_progress(processing_time)
            
            tracker.finish_lambda()
        
        tracker.finish_dataset()
    
    tracker.finish_experiment()
    
    # 最終統計表示
    final_stats = tracker.get_current_stats()
    print(f"\n📋 最終統計:")
    print(f"  完了率: {final_stats.progress_percent:.1f}%")
    print(f"  処理速度: {final_stats.samples_per_second:.2f} samples/sec")
    print(f"  平均処理時間: {final_stats.avg_processing_time:.2f}秒")