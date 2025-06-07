#!/usr/bin/env python3
"""
Comprehensive Experiment Logger for Adaptive Speculative Decoding Research
完全な実験環境・結果記録システム

論文執筆時の参考資料として、すべての実験情報を一元管理
"""

import json
import yaml
import logging
import datetime
import platform
import psutil
import torch
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
# 相対インポートは後で設定
# from .raw_data_logger import RawDataLogger, SampleResult

@dataclass
class ExperimentEnvironment:
    """実験環境の完全記録"""
    timestamp: str
    experiment_id: str
    git_commit: str
    git_branch: str
    
    # ハードウェア環境
    hardware: Dict[str, Any]
    
    # ソフトウェア環境  
    software: Dict[str, Any]
    
    # モデル設定
    models: Dict[str, Any]
    
    # データセット設定
    datasets: Dict[str, Any]
    
    # 実験パラメータ
    parameters: Dict[str, Any]

@dataclass  
class ExperimentResults:
    """実験結果の完全記録"""
    timestamp: str
    experiment_id: str
    
    # メイン結果
    main_results: Dict[str, Any]
    
    # アブレーション研究
    ablation_studies: Dict[str, Any]
    
    # 統計分析
    statistical_analysis: Dict[str, Any]
    
    # パフォーマンス測定
    performance_metrics: Dict[str, Any]
    
    # エラー・警告
    errors_warnings: List[str]

class ComprehensiveLogger:
    """
    論文執筆用の包括的実験ログシステム
    
    beginner_guide_japanese.texの形式を参考に、
    実験環境の完全再現と結果の詳細記録を実現
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "./logs/"):
        self.experiment_name = experiment_name
        self.experiment_id = f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログファイルパス
        self.log_file = self.output_dir / f"{self.experiment_id}_comprehensive.md"
        self.json_file = self.output_dir / f"{self.experiment_id}_data.json"
        
        # 環境・結果記録
        self.environment: Optional[ExperimentEnvironment] = None
        self.results: Optional[ExperimentResults] = None
        
        # 生データロガー（統合済み）
        try:
            from .raw_data_logger import RawDataLogger
            self.raw_data_logger = RawDataLogger(
                self.experiment_id, 
                str(self.output_dir / "raw_data"),
                enable_progress_tracking=False  # 包括ログとの重複を避ける
            )
        except ImportError:
            self.raw_data_logger = None
        
        # ログ開始
        self._initialize_logging()
        
    def _initialize_logging(self):
        """ログシステム初期化"""
        # Markdownヘッダー作成
        header = f"""# 適応的推測デコーディング実験記録
## 実験ID: {self.experiment_id}
## 実験名: {self.experiment_name}
## 作成日時: {datetime.datetime.now().isoformat()}

---

**このファイルについて:**
- 本ファイルは実験環境の完全再現と結果の詳細記録を目的とする
- 論文執筆時の参考資料として設計
- すべての実験情報が一元管理されている

---

"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def capture_environment(self, config_files: List[str] = None) -> ExperimentEnvironment:
        """
        実験環境の完全キャプチャ
        
        Args:
            config_files: 記録する設定ファイルのリスト
        """
        print("🔍 実験環境をキャプチャ中...")
        
        # Git情報取得
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        except:
            git_commit = "unknown"
            git_branch = "unknown"
        
        # ハードウェア情報
        hardware = {
            "cpu": {
                "model": platform.processor(),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            },
            "gpu": self._get_gpu_info(),
            "storage": self._get_storage_info()
        }
        
        # ソフトウェア情報
        software = {
            "os": {
                "system": platform.system(),
                "version": platform.version(),
                "release": platform.release()
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable
            },
            "packages": self._get_package_versions()
        }
        
        # 設定ファイル読み込み
        models_config = self._load_config_file("configs/qwen2.5_models.yaml")
        datasets_config = self._load_config_file("configs/evaluation.yaml")
        
        # 追加の詳細情報取得
        additional_details = self._get_additional_experiment_details()
        
        # 実験パラメータ
        parameters = {
            "lambda_values": models_config.get("experiment", {}).get("lambda_values", []),
            "num_seeds": models_config.get("experiment", {}).get("num_seeds", 5),
            "confidence_level": models_config.get("experiment", {}).get("confidence_level", 0.95)
        }
        
        self.environment = ExperimentEnvironment(
            timestamp=datetime.datetime.now().isoformat(),
            experiment_id=self.experiment_id,
            git_commit=git_commit,
            git_branch=git_branch,
            hardware=hardware,
            software=software,
            models=models_config,
            datasets=datasets_config,
            parameters=parameters
        )
        
        self._write_environment_section()
        print("✅ 実験環境キャプチャ完了")
        return self.environment
    
    def _get_gpu_info(self) -> List[Dict]:
        """GPU情報取得"""
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / (1024**3), 2),
                    "compute_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                })
        return gpu_info
    
    def _get_storage_info(self) -> Dict:
        """ストレージ情報取得"""
        storage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                storage[partition.mountpoint] = {
                    "total_gb": round(usage.total / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "used_percent": round((usage.used / usage.total) * 100, 1)
                }
            except:
                pass
        return storage
    
    def _get_package_versions(self) -> Dict:
        """主要パッケージのバージョン取得"""
        packages = {}
        important_packages = ['torch', 'transformers', 'vllm', 'numpy', 'pandas', 'scikit-learn']
        
        for pkg in important_packages:
            try:
                package = __import__(pkg)
                packages[pkg] = getattr(package, '__version__', 'unknown')
            except ImportError:
                packages[pkg] = 'not_installed'
        
        return packages
    
    def _get_additional_experiment_details(self) -> Dict:
        """追加の実験詳細情報取得"""
        details = {}
        
        # モデルファイルの存在確認
        model_verification = {}
        model_paths = [
            "/raid/sasaki/adaptive-sd-models/qwen2.5-7b/",
            "/raid/sasaki/adaptive-sd-models/qwen2.5-14b/", 
            "/raid/sasaki/adaptive-sd-models/qwen2.5-32b/",
            "/raid/sasaki/adaptive-sd-models/qwen2.5-72b/"
        ]
        
        for path in model_paths:
            model_name = Path(path).name
            if Path(path).exists():
                safetensors_files = list(Path(path).glob("*.safetensors"))
                config_files = list(Path(path).glob("config.json"))
                model_verification[model_name] = {
                    "path": path,
                    "exists": True,
                    "safetensors_count": len(safetensors_files),
                    "has_config": len(config_files) > 0,
                    "total_size_gb": self._get_directory_size(path)
                }
            else:
                model_verification[model_name] = {
                    "path": path,
                    "exists": False
                }
        
        details["model_verification"] = model_verification
        
        # データセット詳細
        dataset_info = {
            "mmlu": {"expected_samples": 14042, "type": "multiple_choice", "subjects": 57},
            "gsm8k": {"expected_samples": 1319, "type": "math_reasoning", "grade_level": "K-8"},
            "humaneval": {"expected_samples": 164, "type": "code_generation", "language": "python"},
            "truthfulqa": {"expected_samples": 817, "type": "truthfulness", "categories": 38}
        }
        details["dataset_info"] = dataset_info
        
        # 実験環境の検証
        environment_checks = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "free_disk_space_gb": round(psutil.disk_usage('.').free / (1024**3), 2),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        }
        details["environment_checks"] = environment_checks
        
        return details
    
    def _get_directory_size(self, path: str) -> float:
        """ディレクトリサイズを取得（GB単位）"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return round(total_size / (1024**3), 2)
        except:
            return 0.0
    
    def _load_config_file(self, filepath: str) -> Dict:
        """設定ファイル読み込み"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    return yaml.safe_load(f)
                elif filepath.endswith('.json'):
                    return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load {filepath}: {str(e)}"}
        return {}
    
    def _write_environment_section(self):
        """環境情報をMarkdownに記録"""
        if not self.environment:
            return
            
        content = f"""
## 1. 実験環境

### 1.1 基本情報
- **実験ID**: {self.environment.experiment_id}
- **実行日時**: {self.environment.timestamp}
- **Git コミット**: `{self.environment.git_commit}`
- **Git ブランチ**: `{self.environment.git_branch}`

### 1.2 ハードウェア構成

#### CPU
- **モデル**: {self.environment.hardware['cpu']['model']}
- **物理コア数**: {self.environment.hardware['cpu']['cores']}
- **論理コア数**: {self.environment.hardware['cpu']['threads']}

#### メモリ
- **総容量**: {self.environment.hardware['memory']['total_gb']} GB
- **使用可能**: {self.environment.hardware['memory']['available_gb']} GB

#### GPU構成
"""
        
        for gpu in self.environment.hardware['gpu']:
            content += f"""
- **GPU {gpu['id']}**: {gpu['name']}
  - メモリ: {gpu['memory_total_gb']} GB
  - Compute Capability: {gpu['compute_capability']}
"""
        
        content += f"""
### 1.3 ソフトウェア環境

#### オペレーティングシステム
- **OS**: {self.environment.software['os']['system']} {self.environment.software['os']['release']}

#### Python環境
- **バージョン**: {self.environment.software['python']['version'].split()[0]}
- **実行パス**: `{self.environment.software['python']['executable']}`

#### 主要パッケージ
"""
        
        for pkg, version in self.environment.software['packages'].items():
            content += f"- **{pkg}**: {version}\n"
        
        content += f"""
### 1.4 モデル設定

#### Qwen2.5 4段階階層
"""
        
        for stage in self.environment.models.get('models', {}).get('stages', []):
            content += f"""
- **{stage['name']}**:
  - パス: `{stage['model_path']}`
  - GPU配置: {stage['gpu_ids']}
  - 並列度: {stage['tensor_parallel_size']}
  - 実測コスト: {stage['relative_cost']}x
  - 実測レイテンシ: {stage['base_latency_ms']}ms
"""
        
        content += f"""
### 1.5 データセット設定

#### 評価データセット（完全版使用）
"""
        
        for dataset in self.environment.datasets.get('datasets', {}).values():
            if isinstance(dataset, dict) and 'max_samples' in dataset:
                content += f"- **{dataset.get('path', 'Unknown')}**: {dataset['max_samples']:,} サンプル\n"
        
        content += f"""
### 1.6 実験パラメータ

- **Lambda値**: {self.environment.parameters['lambda_values']}
- **試行回数**: {self.environment.parameters['num_seeds']}
- **信頼水準**: {self.environment.parameters['confidence_level']}

---

"""
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def record_results(self, main_results: Dict, ablation_studies: Dict = None, 
                      statistical_analysis: Dict = None, performance_metrics: Dict = None,
                      errors_warnings: List[str] = None, raw_data: Dict = None):
        """
        実験結果の詳細記録
        
        Args:
            main_results: メインの実験結果
            ablation_studies: アブレーション研究結果
            statistical_analysis: 統計分析結果
            performance_metrics: パフォーマンス測定結果
            errors_warnings: エラー・警告のリスト
        """
        print("📊 実験結果を記録中...")
        
        self.results = ExperimentResults(
            timestamp=datetime.datetime.now().isoformat(),
            experiment_id=self.experiment_id,
            main_results=main_results or {},
            ablation_studies=ablation_studies or {},
            statistical_analysis=statistical_analysis or {},
            performance_metrics=performance_metrics or {},
            errors_warnings=errors_warnings or []
        )
        
        self._write_results_section()
        print("✅ 実験結果記録完了")
    
    def _write_results_section(self):
        """結果情報をMarkdownに記録"""
        if not self.results:
            return
            
        content = f"""
## 2. 実験結果

### 2.1 メイン結果

#### 全体性能サマリー
"""
        
        # メイン結果の記録
        if 'summary' in self.results.main_results:
            summary = self.results.main_results['summary']
            content += f"""
- **平均高速化**: {summary.get('avg_speedup', 'N/A')}倍
- **品質保持率**: {summary.get('quality_retention', 'N/A')}%
- **総処理サンプル数**: {summary.get('total_samples', 'N/A'):,}
- **実験完了時刻**: {self.results.timestamp}
"""
        
        # 生データの詳細記録
        if 'raw_data_summary' in self.results.main_results:
            raw_summary = self.results.main_results['raw_data_summary']
            content += f"""

#### 生データ詳細サマリー
- **記録されたサンプル総数**: {raw_summary.get('total_samples_recorded', 'N/A'):,}
- **平均処理時間/サンプル**: {raw_summary.get('avg_time_per_sample', 'N/A'):.3f}秒
- **段階別使用分布**:
"""
            if 'stage_usage_distribution' in raw_summary:
                for stage, count in raw_summary['stage_usage_distribution'].items():
                    percentage = (count / raw_summary.get('total_samples_recorded', 1)) * 100
                    content += f"  - Stage {stage}: {count:,}回 ({percentage:.1f}%)\n"
            
            content += f"""
- **レイテンシ詳細**:
  - 最小: {raw_summary.get('latency_min', 'N/A'):.2f}ms
  - 最大: {raw_summary.get('latency_max', 'N/A'):.2f}ms  
  - 中央値: {raw_summary.get('latency_median', 'N/A'):.2f}ms
  - 標準偏差: {raw_summary.get('latency_std', 'N/A'):.2f}ms
"""
        
        # データセット別結果
        if 'by_dataset' in self.results.main_results:
            content += "\n#### データセット別結果\n\n"
            content += "| データセット | サンプル数 | 精度 | 高速化 | 品質保持 |\n"
            content += "|-------------|-----------|------|--------|----------|\n"
            
            for dataset, metrics in self.results.main_results['by_dataset'].items():
                content += f"| {dataset} | {metrics.get('samples', 'N/A'):,} | {metrics.get('accuracy', 'N/A'):.3f} | {metrics.get('speedup', 'N/A'):.2f}x | {metrics.get('quality', 'N/A'):.1f}% |\n"
        
        # Lambda値別結果
        if 'by_lambda' in self.results.main_results:
            content += "\n#### Lambda値別性能\n\n"
            content += "| Lambda | 高速化 | 品質 | 早期停止率 |\n"
            content += "|--------|--------|------|------------|\n"
            
            for lambda_val, metrics in self.results.main_results['by_lambda'].items():
                content += f"| {lambda_val} | {metrics.get('speedup', 'N/A'):.2f}x | {metrics.get('quality', 'N/A'):.3f} | {metrics.get('early_stop_rate', 'N/A'):.1f}% |\n"
        
        # アブレーション研究
        if self.results.ablation_studies:
            content += "\n### 2.2 アブレーション研究（要素分析）\n\n"
            
            for component, results in self.results.ablation_studies.items():
                content += f"#### {component}の影響\n\n"
                if isinstance(results, dict):
                    for metric, value in results.items():
                        content += f"- **{metric}**: {value}\n"
                content += "\n"
        
        # 統計分析
        if self.results.statistical_analysis:
            content += "\n### 2.3 統計的有意性検定\n\n"
            
            if 'significance_tests' in self.results.statistical_analysis:
                content += "#### 有意性検定結果\n\n"
                for test_name, result in self.results.statistical_analysis['significance_tests'].items():
                    content += f"- **{test_name}**: p値 = {result.get('p_value', 'N/A')}, 有意 = {result.get('significant', 'N/A')}\n"
            
            if 'confidence_intervals' in self.results.statistical_analysis:
                content += "\n#### 信頼区間\n\n"
                for metric, ci in self.results.statistical_analysis['confidence_intervals'].items():
                    content += f"- **{metric}**: [{ci.get('lower', 'N/A'):.3f}, {ci.get('upper', 'N/A'):.3f}] (95%信頼区間)\n"
        
        # パフォーマンス測定
        if self.results.performance_metrics:
            content += "\n### 2.4 詳細パフォーマンス測定\n\n"
            
            if 'latency' in self.results.performance_metrics:
                content += "#### レイテンシ測定\n\n"
                latency = self.results.performance_metrics['latency']
                content += f"- **平均レイテンシ**: {latency.get('mean', 'N/A'):.2f}ms\n"
                content += f"- **標準偏差**: {latency.get('std', 'N/A'):.2f}ms\n"
                content += f"- **P95レイテンシ**: {latency.get('p95', 'N/A'):.2f}ms\n"
            
            if 'throughput' in self.results.performance_metrics:
                content += "\n#### スループット測定\n\n"
                throughput = self.results.performance_metrics['throughput']
                content += f"- **平均スループット**: {throughput.get('mean', 'N/A'):.2f} tokens/sec\n"
                content += f"- **ピークスループット**: {throughput.get('peak', 'N/A'):.2f} tokens/sec\n"
        
        # エラー・警告
        if self.results.errors_warnings:
            content += "\n### 2.5 エラー・警告\n\n"
            for i, error in enumerate(self.results.errors_warnings, 1):
                content += f"{i}. {error}\n"
        
        content += "\n---\n\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def add_section(self, title: str, content: str):
        """カスタムセクション追加"""
        section_content = f"""
## {title}

{content}

---

"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(section_content)
    
    def save_data(self):
        """構造化データをJSONで保存"""
        data = {
            "environment": asdict(self.environment) if self.environment else None,
            "results": asdict(self.results) if self.results else None
        }
        
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def finalize(self):
        """ログの最終化"""
        # 最終セクション追加
        final_content = f"""
## 3. 論文執筆用情報

### 3.1 再現性のための重要情報

本実験は以下の環境で実行されました：

- **ハードウェア**: {len(self.environment.hardware['gpu']) if self.environment else 'N/A'}x GPU構成
- **モデル**: Qwen2.5 4段階階層（7B→14B→32B→72B）
- **データセット**: 完全版（総計{sum(d.get('max_samples', 0) for d in self.environment.datasets.get('datasets', {}).values() if isinstance(d, dict)) if self.environment else 'N/A':,}サンプル）
- **実験パラメータ**: Lambda値6点、シード5回

### 3.2 主要な知見

1. **効率性**: 平均{self.results.main_results.get('summary', {}).get('avg_speedup', 'N/A')}倍の高速化を達成
2. **品質**: {self.results.main_results.get('summary', {}).get('quality_retention', 'N/A')}%の品質保持を実現
3. **理論的保証**: 最適停止理論による数学的裏付け

### 3.3 ファイル情報

- **包括ログ**: `{self.log_file.name}`
- **構造化データ**: `{self.json_file.name}`
- **実験ID**: `{self.experiment_id}`

---

**実験完了**: {datetime.datetime.now().isoformat()}

本ログファイルは論文執筆時の参考資料として作成されました。
すべての実験詳細が記録されており、完全な再現が可能です。
"""
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(final_content)
        
        # JSONデータ保存
        self.save_data()
        
        print(f"📝 包括ログ作成完了:")
        print(f"   📄 Markdown: {self.log_file}")
        print(f"   💾 JSON Data: {self.json_file}")
        print(f"   🆔 Experiment ID: {self.experiment_id}")

# 使用例
if __name__ == "__main__":
    # ログシステム初期化
    logger = ComprehensiveLogger("full_evaluation_test")
    
    # 環境キャプチャ
    logger.capture_environment()
    
    # サンプル結果記録
    sample_results = {
        "summary": {
            "avg_speedup": 3.6,
            "quality_retention": 91.2,
            "total_samples": 16342
        },
        "by_dataset": {
            "MMLU": {"samples": 14042, "accuracy": 0.852, "speedup": 3.2, "quality": 90.5},
            "GSM8K": {"samples": 1319, "accuracy": 0.743, "speedup": 4.1, "quality": 92.8}
        }
    }
    
    ablation_results = {
        "quality_predictor": {
            "with_predictor": 3.6,
            "without_predictor": 1.8,
            "improvement": "2.0x"
        }
    }
    
    logger.record_results(sample_results, ablation_results)
    logger.finalize()