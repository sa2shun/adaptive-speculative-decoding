#!/usr/bin/env python3
"""
生データ詳細記録システム
Raw Data Detailed Logging System

実際の実験データをすべて記録 - 数値の要約ではなく生の測定値
"""

import json
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import datetime
# 相対インポートは後で設定
# from .progress_tracker import ExperimentProgressTracker

@dataclass
class SampleResult:
    """個別サンプルの結果"""
    sample_id: str
    dataset: str
    input_text: str
    ground_truth: str
    
    # 各段階の詳細記録
    stage_outputs: List[str]  # 各段階の出力
    stage_latencies: List[float]  # 各段階の実測レイテンシ (ms)
    stage_tokens: List[int]  # 各段階の生成トークン数
    stage_probabilities: List[float]  # 各段階の品質予測確率
    
    # 停止判断の詳細
    stopped_at_stage: int
    stop_reason: str  # "quality_threshold" | "final_stage" | "error"
    quality_score: float
    threshold_used: float
    
    # 最終結果
    final_output: str
    is_correct: bool
    evaluation_score: float
    
    # タイミング詳細
    total_latency: float
    first_token_latency: float
    generation_latency: float
    overhead_latency: float
    
    # リソース使用量
    gpu_memory_used: List[float]  # 各段階のGPUメモリ使用量
    cpu_usage: float
    
    # Lambda実験用
    lambda_value: float
    seed: int

@dataclass
class DetailedExperimentData:
    """詳細実験データ"""
    experiment_id: str
    timestamp: str
    
    # 全サンプルの生データ
    all_samples: List[SampleResult]
    
    # 時系列データ
    timing_series: List[Dict]  # タイムスタンプ付きイベント
    resource_series: List[Dict]  # リソース使用量の時系列
    
    # 段階別統計（生データから計算）
    stage_statistics: Dict[int, Dict]
    
    # データセット別生データ
    dataset_raw_results: Dict[str, List[SampleResult]]
    
    # Lambda値別生データ
    lambda_raw_results: Dict[float, List[SampleResult]]

class RawDataLogger:
    """
    実データ詳細記録システム
    
    すべてのサンプル、すべての測定値、すべてのタイミングを記録
    """
    
    def __init__(self, experiment_id: str, output_dir: str = "./logs/raw_data/", 
                 enable_progress_tracking: bool = True, 
                 datasets: List[str] = None, lambda_values: List[float] = None,
                 total_samples: int = None):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ格納
        self.all_samples: List[SampleResult] = []
        self.timing_events: List[Dict] = []
        self.resource_snapshots: List[Dict] = []
        
        # 進捗トラッキング
        self.enable_progress_tracking = enable_progress_tracking
        self.progress_tracker = None
        if enable_progress_tracking and datasets and lambda_values and total_samples:
            try:
                from .progress_tracker import ExperimentProgressTracker
            except ImportError:
                try:
                    from progress_tracker import ExperimentProgressTracker
                except ImportError:
                    print("⚠️ 進捗トラッカーをインポートできませんでした。進捗表示は無効です。")
                    self.enable_progress_tracking = False
                    ExperimentProgressTracker = None
            
            if ExperimentProgressTracker:
                self.progress_tracker = ExperimentProgressTracker(
                    total_samples=total_samples,
                    datasets=datasets,
                    lambda_values=lambda_values
                )
        
        # ファイルパス
        self.samples_file = self.output_dir / f"{experiment_id}_all_samples.json"
        self.timing_file = self.output_dir / f"{experiment_id}_timing_series.csv"
        self.resource_file = self.output_dir / f"{experiment_id}_resource_series.csv"
        self.summary_file = self.output_dir / f"{experiment_id}_raw_summary.json"
        
        # 実験開始記録
        self.log_event("experiment_start", {"experiment_id": experiment_id})
    
    def log_sample_result(self, 
                         sample_id: str,
                         dataset: str,
                         input_text: str,
                         ground_truth: str,
                         stage_outputs: List[str],
                         stage_latencies: List[float],
                         stage_tokens: List[int],
                         stage_probabilities: List[float],
                         stopped_at_stage: int,
                         stop_reason: str,
                         quality_score: float,
                         threshold_used: float,
                         final_output: str,
                         is_correct: bool,
                         evaluation_score: float,
                         gpu_memory_used: List[float],
                         cpu_usage: float,
                         lambda_value: float,
                         seed: int):
        """個別サンプル結果の詳細記録"""
        
        # タイミング計算
        total_latency = sum(stage_latencies)
        first_token_latency = stage_latencies[0] if stage_latencies else 0.0
        generation_latency = sum(stage_latencies[1:]) if len(stage_latencies) > 1 else 0.0
        overhead_latency = 0.0  # 実装時に測定
        
        sample_result = SampleResult(
            sample_id=sample_id,
            dataset=dataset,
            input_text=input_text,
            ground_truth=ground_truth,
            stage_outputs=stage_outputs,
            stage_latencies=stage_latencies,
            stage_tokens=stage_tokens,
            stage_probabilities=stage_probabilities,
            stopped_at_stage=stopped_at_stage,
            stop_reason=stop_reason,
            quality_score=quality_score,
            threshold_used=threshold_used,
            final_output=final_output,
            is_correct=is_correct,
            evaluation_score=evaluation_score,
            total_latency=total_latency,
            first_token_latency=first_token_latency,
            generation_latency=generation_latency,
            overhead_latency=overhead_latency,
            gpu_memory_used=gpu_memory_used,
            cpu_usage=cpu_usage,
            lambda_value=lambda_value,
            seed=seed
        )
        
        self.all_samples.append(sample_result)
        
        # 進捗更新
        if self.progress_tracker:
            processing_time = total_latency / 1000.0  # ms to seconds
            self.progress_tracker.update_progress(processing_time)
        
        # リアルタイム保存（大規模実験での安全性確保）
        if len(self.all_samples) % 100 == 0:
            self._save_samples_batch()
        
        return sample_result
    
    def log_event(self, event_type: str, details: Dict):
        """タイミングイベント記録"""
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.timing_events.append(event)
    
    def log_resource_snapshot(self, gpu_memory: List[float], gpu_utilization: List[float], 
                            cpu_usage: float, system_memory: float):
        """リソース使用量スナップショット"""
        snapshot = {
            "timestamp": datetime.datetime.now().isoformat(),
            "gpu_memory_gb": gpu_memory,
            "gpu_utilization_percent": gpu_utilization,
            "cpu_usage_percent": cpu_usage,
            "system_memory_gb": system_memory
        }
        self.resource_snapshots.append(snapshot)
    
    def start_dataset(self, dataset: str, expected_samples: int):
        """データセット開始通知"""
        if self.progress_tracker:
            self.progress_tracker.start_dataset(dataset, expected_samples)
        self.log_event("dataset_start", {"dataset": dataset, "expected_samples": expected_samples})
    
    def start_lambda(self, lambda_value: float, expected_samples: int):
        """Lambda値設定開始通知"""
        if self.progress_tracker:
            self.progress_tracker.start_lambda(lambda_value, expected_samples)
        self.log_event("lambda_start", {"lambda": lambda_value, "expected_samples": expected_samples})
    
    def finish_lambda(self):
        """Lambda値完了通知"""
        if self.progress_tracker:
            self.progress_tracker.finish_lambda()
        self.log_event("lambda_finish", {"lambda": self.progress_tracker.current_lambda if self.progress_tracker else None})
    
    def finish_dataset(self):
        """データセット完了通知"""
        if self.progress_tracker:
            self.progress_tracker.finish_dataset()
        self.log_event("dataset_finish", {"dataset": self.progress_tracker.current_dataset if self.progress_tracker else None})
    
    def get_progress_stats(self):
        """現在の進捗統計取得"""
        if self.progress_tracker:
            return self.progress_tracker.get_current_stats()
        return None
    
    def _save_samples_batch(self):
        """サンプルデータのバッチ保存"""
        samples_data = [asdict(sample) for sample in self.all_samples]
        with open(self.samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)
    
    def generate_detailed_analysis(self) -> DetailedExperimentData:
        """生データから詳細分析生成"""
        
        # データセット別分類
        dataset_results = {}
        for sample in self.all_samples:
            if sample.dataset not in dataset_results:
                dataset_results[sample.dataset] = []
            dataset_results[sample.dataset].append(sample)
        
        # Lambda値別分類
        lambda_results = {}
        for sample in self.all_samples:
            if sample.lambda_value not in lambda_results:
                lambda_results[sample.lambda_value] = []
            lambda_results[sample.lambda_value].append(sample)
        
        # 段階別統計計算
        stage_stats = {}
        for stage in range(4):  # 0, 1, 2, 3
            stage_samples = [s for s in self.all_samples if s.stopped_at_stage >= stage]
            if stage_samples:
                latencies = [s.stage_latencies[stage] for s in stage_samples if len(s.stage_latencies) > stage]
                tokens = [s.stage_tokens[stage] for s in stage_samples if len(s.stage_tokens) > stage]
                probabilities = [s.stage_probabilities[stage] for s in stage_samples if len(s.stage_probabilities) > stage]
                
                stage_stats[stage] = {
                    "sample_count": len(stage_samples),
                    "latency_stats": {
                        "mean": np.mean(latencies) if latencies else 0,
                        "std": np.std(latencies) if latencies else 0,
                        "min": np.min(latencies) if latencies else 0,
                        "max": np.max(latencies) if latencies else 0,
                        "p50": np.percentile(latencies, 50) if latencies else 0,
                        "p95": np.percentile(latencies, 95) if latencies else 0,
                        "p99": np.percentile(latencies, 99) if latencies else 0,
                        "raw_values": latencies  # 生の値も保存
                    },
                    "token_stats": {
                        "mean": np.mean(tokens) if tokens else 0,
                        "std": np.std(tokens) if tokens else 0,
                        "total": sum(tokens) if tokens else 0,
                        "raw_values": tokens
                    },
                    "probability_stats": {
                        "mean": np.mean(probabilities) if probabilities else 0,
                        "std": np.std(probabilities) if probabilities else 0,
                        "raw_values": probabilities
                    }
                }
        
        return DetailedExperimentData(
            experiment_id=self.experiment_id,
            timestamp=datetime.datetime.now().isoformat(),
            all_samples=self.all_samples,
            timing_series=self.timing_events,
            resource_series=self.resource_snapshots,
            stage_statistics=stage_stats,
            dataset_raw_results=dataset_results,
            lambda_raw_results=lambda_results
        )
    
    def save_all_data(self):
        """すべてのデータを保存"""
        
        # 1. 全サンプルデータ
        self._save_samples_batch()
        
        # 2. タイミングデータ (CSV)
        if self.timing_events:
            timing_df = pd.DataFrame(self.timing_events)
            timing_df.to_csv(self.timing_file, index=False)
        
        # 3. リソースデータ (CSV)
        if self.resource_snapshots:
            resource_df = pd.DataFrame(self.resource_snapshots)
            resource_df.to_csv(self.resource_file, index=False)
        
        # 4. 詳細分析データ
        detailed_data = self.generate_detailed_analysis()
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            # dataclassをJSONシリアライズ可能に変換
            data_dict = asdict(detailed_data)
            json.dump(data_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # 実験完了通知
        if self.progress_tracker:
            self.progress_tracker.finish_experiment()
        
        return {
            "samples_file": str(self.samples_file),
            "timing_file": str(self.timing_file),
            "resource_file": str(self.resource_file),
            "summary_file": str(self.summary_file),
            "total_samples": len(self.all_samples)
        }
    
    def generate_raw_data_report(self) -> str:
        """生データレポート生成"""
        
        if not self.all_samples:
            return "データなし"
        
        # データセット別統計
        dataset_stats = {}
        for dataset in set(s.dataset for s in self.all_samples):
            samples = [s for s in self.all_samples if s.dataset == dataset]
            dataset_stats[dataset] = {
                "sample_count": len(samples),
                "accuracy": sum(s.is_correct for s in samples) / len(samples),
                "avg_latency": np.mean([s.total_latency for s in samples]),
                "avg_stages_used": np.mean([s.stopped_at_stage + 1 for s in samples]),
                "latency_distribution": [s.total_latency for s in samples],
                "stage_distribution": [s.stopped_at_stage for s in samples]
            }
        
        # Lambda値別統計
        lambda_stats = {}
        for lambda_val in set(s.lambda_value for s in self.all_samples):
            samples = [s for s in self.all_samples if s.lambda_value == lambda_val]
            lambda_stats[lambda_val] = {
                "sample_count": len(samples),
                "avg_speedup": np.mean([4.0 / (s.stopped_at_stage + 1) for s in samples]),  # 仮の計算
                "early_stop_rate": sum(s.stopped_at_stage < 3 for s in samples) / len(samples),
                "quality_scores": [s.quality_score for s in samples],
                "latencies": [s.total_latency for s in samples]
            }
        
        report = f"""
# 生データ詳細レポート
## 実験ID: {self.experiment_id}

## 全体統計
- **総サンプル数**: {len(self.all_samples):,}
- **データセット数**: {len(dataset_stats)}
- **Lambda設定数**: {len(lambda_stats)}
- **実験期間**: {self.timing_events[0]['timestamp'] if self.timing_events else 'N/A'} ~ {self.timing_events[-1]['timestamp'] if self.timing_events else 'N/A'}

## データセット別生データ
"""
        
        for dataset, stats in dataset_stats.items():
            report += f"""
### {dataset}
- **サンプル数**: {stats['sample_count']:,}
- **精度**: {stats['accuracy']:.4f}
- **平均レイテンシ**: {stats['avg_latency']:.2f}ms
- **平均使用段階数**: {stats['avg_stages_used']:.2f}
- **レイテンシ分布**: min={np.min(stats['latency_distribution']):.2f}, max={np.max(stats['latency_distribution']):.2f}, std={np.std(stats['latency_distribution']):.2f}
- **段階使用分布**: {dict(zip(*np.unique(stats['stage_distribution'], return_counts=True)))}
"""
        
        report += "\n## Lambda値別生データ\n"
        
        for lambda_val, stats in lambda_stats.items():
            report += f"""
### λ = {lambda_val}
- **サンプル数**: {stats['sample_count']:,}
- **平均高速化**: {stats['avg_speedup']:.2f}x
- **早期停止率**: {stats['early_stop_rate']:.1%}
- **品質スコア**: mean={np.mean(stats['quality_scores']):.4f}, std={np.std(stats['quality_scores']):.4f}
- **レイテンシ**: mean={np.mean(stats['latencies']):.2f}, std={np.std(stats['latencies']):.2f}
"""
        
        return report

# 使用例
if __name__ == "__main__":
    # 生データロガー初期化
    logger = RawDataLogger("test_raw_data")
    
    # サンプル実験データ記録
    import random
    
    datasets = ["MMLU", "GSM8K", "HumanEval", "TruthfulQA"]
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for i in range(100):  # 100サンプルのテストデータ
        dataset = random.choice(datasets)
        lambda_val = random.choice(lambda_values)
        
        # 模擬データ生成
        stage_latencies = [
            random.uniform(800, 1200),  # Stage 0
            random.uniform(1600, 2400),  # Stage 1  
            random.uniform(3200, 4800),  # Stage 2
            random.uniform(6400, 9600)   # Stage 3
        ]
        
        stopped_stage = random.randint(0, 3)
        used_latencies = stage_latencies[:stopped_stage + 1]
        
        logger.log_sample_result(
            sample_id=f"{dataset}_{i:04d}",
            dataset=dataset,
            input_text=f"Sample input text {i}",
            ground_truth=f"Ground truth {i}",
            stage_outputs=[f"Output stage {j}" for j in range(stopped_stage + 1)],
            stage_latencies=used_latencies,
            stage_tokens=[random.randint(50, 200) for _ in range(stopped_stage + 1)],
            stage_probabilities=[random.uniform(0.7, 0.95) for _ in range(stopped_stage + 1)],
            stopped_at_stage=stopped_stage,
            stop_reason="quality_threshold" if stopped_stage < 3 else "final_stage",
            quality_score=random.uniform(0.8, 0.95),
            threshold_used=random.uniform(0.75, 0.9),
            final_output=f"Final output {i}",
            is_correct=random.choice([True, False]),
            evaluation_score=random.uniform(0.0, 1.0),
            gpu_memory_used=[random.uniform(10, 80) for _ in range(stopped_stage + 1)],
            cpu_usage=random.uniform(20, 60),
            lambda_value=lambda_val,
            seed=42
        )
    
    # データ保存
    file_info = logger.save_all_data()
    print("📊 生データ保存完了:")
    for key, path in file_info.items():
        print(f"  {key}: {path}")
    
    # レポート生成
    report = logger.generate_raw_data_report()
    print("\n" + "="*50)
    print(report)