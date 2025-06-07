#!/usr/bin/env python3
"""
実験フック：既存の実験スクリプトに包括ログを統合

既存の実験コードを最小限の変更で包括ログシステムに対応
"""

import functools
import time
import traceback
from typing import Callable, Any, Dict, Optional
from pathlib import Path

try:
    from .comprehensive_logger import ComprehensiveLogger
except ImportError:
    from comprehensive_logger import ComprehensiveLogger

class ExperimentTracker:
    """
    既存実験への包括ログ統合
    
    デコレータとコンテキストマネージャーで
    既存コードに最小限の変更で統合
    """
    
    _current_logger: Optional[ComprehensiveLogger] = None
    _experiment_data: Dict[str, Any] = {}
    
    @classmethod
    def initialize(cls, experiment_name: str) -> ComprehensiveLogger:
        """実験トラッカー初期化"""
        cls._current_logger = ComprehensiveLogger(experiment_name)
        cls._experiment_data = {
            "start_time": time.time(),
            "results": {},
            "errors": [],
            "performance": {}
        }
        
        # 環境キャプチャ
        cls._current_logger.capture_environment()
        
        return cls._current_logger
    
    @classmethod
    def get_logger(cls) -> Optional[ComprehensiveLogger]:
        """現在のロガー取得"""
        return cls._current_logger
    
    @classmethod
    def record_result(cls, section: str, data: Any):
        """結果記録"""
        if cls._current_logger:
            cls._experiment_data["results"][section] = data
    
    @classmethod
    def record_error(cls, error: str):
        """エラー記録"""
        cls._experiment_data["errors"].append(error)
    
    @classmethod
    def record_performance(cls, metric: str, value: Any):
        """パフォーマンス記録"""
        cls._experiment_data["performance"][metric] = value
    
    @classmethod
    def finalize(cls):
        """実験完了処理"""
        if cls._current_logger:
            # 実行時間計算
            total_time = time.time() - cls._experiment_data["start_time"]
            cls._experiment_data["performance"]["total_runtime_seconds"] = total_time
            
            # 結果記録
            cls._current_logger.record_results(
                main_results=cls._experiment_data["results"],
                performance_metrics=cls._experiment_data["performance"],
                errors_warnings=cls._experiment_data["errors"]
            )
            
            # ログ完成
            cls._current_logger.finalize()
            
            return cls._current_logger.experiment_id
        
        return None

def track_experiment(experiment_name: str):
    """
    実験トラッキングデコレータ
    
    Usage:
        @track_experiment("my_experiment")
        def run_my_experiment():
            # 実験コード
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 実験開始
            logger = ExperimentTracker.initialize(experiment_name)
            print(f"🚀 実験開始: {experiment_name}")
            print(f"📝 ログID: {logger.experiment_id}")
            
            try:
                # 実験実行
                result = func(*args, **kwargs)
                
                # 正常完了
                print("✅ 実験正常完了")
                experiment_id = ExperimentTracker.finalize()
                print(f"📄 ログ保存: {logger.log_file}")
                
                return result
                
            except Exception as e:
                # エラー処理
                error_msg = f"実験エラー: {str(e)}"
                print(f"❌ {error_msg}")
                
                ExperimentTracker.record_error(error_msg)
                ExperimentTracker.record_error(traceback.format_exc())
                ExperimentTracker.finalize()
                
                raise
        
        return wrapper
    return decorator

def log_section(section_name: str):
    """
    セクション記録デコレータ
    
    Usage:
        @log_section("dataset_evaluation")
        def evaluate_datasets():
            # 評価コード
            return results
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"📊 {section_name} 開始...")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # 実行時間記録
                duration = time.time() - start_time
                ExperimentTracker.record_performance(f"{section_name}_duration", duration)
                
                # 結果記録
                if result is not None:
                    ExperimentTracker.record_result(section_name, result)
                
                print(f"✅ {section_name} 完了 ({duration:.1f}秒)")
                return result
                
            except Exception as e:
                error_msg = f"{section_name}でエラー: {str(e)}"
                ExperimentTracker.record_error(error_msg)
                print(f"❌ {error_msg}")
                raise
        
        return wrapper
    return decorator

class ExperimentContext:
    """
    実験コンテキストマネージャー
    
    Usage:
        with ExperimentContext("my_experiment") as ctx:
            # 実験コード
            ctx.log_result("accuracy", 0.95)
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.logger: Optional[ComprehensiveLogger] = None
    
    def __enter__(self):
        self.logger = ExperimentTracker.initialize(self.experiment_name)
        print(f"🚀 実験開始: {self.experiment_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # エラー発生
            error_msg = f"実験エラー: {str(exc_val)}"
            ExperimentTracker.record_error(error_msg)
            print(f"❌ {error_msg}")
        
        # 完了処理
        experiment_id = ExperimentTracker.finalize()
        if self.logger:
            print(f"📄 ログ保存: {self.logger.log_file}")
        
        return False  # 例外を再発生
    
    def log_result(self, key: str, value: Any):
        """結果記録"""
        ExperimentTracker.record_result(key, value)
    
    def log_performance(self, metric: str, value: Any):
        """パフォーマンス記録"""
        ExperimentTracker.record_performance(metric, value)
    
    def log_section(self, section: str, data: Dict):
        """セクション記録"""
        ExperimentTracker.record_result(section, data)

# 便利関数
def quick_log(key: str, value: Any):
    """クイック結果記録"""
    ExperimentTracker.record_result(key, value)

def log_metric(metric: str, value: Any):
    """メトリクス記録"""
    ExperimentTracker.record_performance(metric, value)

def log_error(error: str):
    """エラー記録"""
    ExperimentTracker.record_error(error)

# 使用例とテスト
if __name__ == "__main__":
    import random
    import numpy as np
    
    # 例1: デコレータ使用
    @track_experiment("decorator_test")
    def test_with_decorator():
        
        @log_section("data_preparation")
        def prepare_data():
            time.sleep(1)  # 処理のシミュレート
            return {"samples": 1000, "features": 512}
        
        @log_section("model_training")  
        def train_model():
            time.sleep(2)
            return {"accuracy": 0.95, "loss": 0.12}
        
        @log_section("evaluation")
        def evaluate():
            time.sleep(1)
            return {"test_accuracy": 0.93, "f1_score": 0.91}
        
        # 実行
        data_info = prepare_data()
        training_results = train_model() 
        eval_results = evaluate()
        
        # 追加メトリクス
        log_metric("total_parameters", 1000000)
        log_metric("training_time", 3600)
        
        return {
            "final_accuracy": eval_results["test_accuracy"],
            "status": "success"
        }
    
    # 例2: コンテキストマネージャー使用
    def test_with_context():
        with ExperimentContext("context_test") as ctx:
            
            # データ準備
            ctx.log_section("preparation", {
                "dataset": "MMLU",
                "samples": 14042
            })
            
            # 評価実行
            results = {
                "accuracy": 0.852,
                "speedup": 3.6,
                "quality": 91.2
            }
            ctx.log_result("main_results", results)
            
            # パフォーマンス記録
            ctx.log_performance("latency_ms", 847.3)
            ctx.log_performance("throughput_tps", 47.2)
    
    print("🧪 包括ログシステムテスト開始\n")
    
    # テスト1
    print("=" * 50)
    print("テスト1: デコレータ使用")
    print("=" * 50)
    test_with_decorator()
    
    print("\n" + "=" * 50)
    print("テスト2: コンテキストマネージャー使用")
    print("=" * 50)
    test_with_context()
    
    print("\n🎉 テスト完了！")