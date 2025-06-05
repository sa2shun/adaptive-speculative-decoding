# 研究実装プロトコル：Adaptive Speculative Decoding

## 1. 研究の全体像と目的

### 1.1 研究目標
- **What**: 8B→13B→34B→70Bの階層Draft-Verifyパイプラインで入力依存の動的停止を実現
- **Why**: 入力難易度に応じた計算資源の最適配分により、品質を維持しつつ推論コストを削減
- **How**: 軽量品質予測モデルと動的計画法による最適停止則の組み合わせ

### 1.2 新規性の明確化
```
既存研究の限界:
- PipeSpec: 深度固定（static-pipeline limitation）
- BanditSpec: モデル階層は固定、ハイパラメータのみ適応
- ML-SpecQD: 最終段verify必須
- DEL: 単一モデル内の層間early exit

本研究の貢献:
→ モデル間階層での動的停止を初めて体系化
```

## 2. 理論的基礎の詳細

### 2.1 問題設定

入力プロンプトx、L段のモデル階層 M = {M₁, M₂, ..., M_L} において：

```
目的関数: min_{k(x)} J(k) = E[Σᵢ₌₁^{k(x)} cᵢ] + λ E[1 - p̄_{k(x)}]

ただし:
- k(x): 入力xに対する停止段
- cᵢ: 段iの計算コスト（FLOPs/token）
- pᵢ: 段iの出力が最終応答として合格する確率
- p̄ᵢ = Πⱼ≤ᵢ pⱼ: 累積合格確率
- λ: 品質重み（Lagrange乗数）
```

### 2.2 動的計画法による最適解

Bellman方程式：
```
Jᵢ = min{
    cᵢ + λ(1 - p̄ᵢ),     # 段iで停止
    cᵢ + Jᵢ₊₁            # 次段へ継続
}
```

後退帰納法によるO(L)アルゴリズム：
```python
def optimal_stopping_rule(p, C, lam):
    """
    Args:
        p: [p₁, p₂, ..., p_L] 各段の合格確率
        C: [c₁, c₂, ..., c_L] 各段のコスト
        lam: 品質重みλ
    Returns:
        k_star: 最適停止段
        J: 各段からの最小期待コスト
    """
    L = len(C)
    J = [0] * (L + 1)  # J[L+1] = 0 (境界条件)
    policy = [False] * L
    
    # 累積確率の事前計算
    p_bar = [1.0]
    for i in range(L):
        p_bar.append(p_bar[-1] * p[i])
    
    # 後退帰納
    for i in reversed(range(L)):
        cost_stop = C[i] + lam * (1 - p_bar[i+1])
        cost_continue = C[i] + J[i+1]
        
        if cost_stop <= cost_continue:
            policy[i] = True  # 停止
            J[i] = cost_stop
        else:
            policy[i] = False  # 継続
            J[i] = cost_continue
    
    # 最初の停止点を返す
    k_star = next((i for i, stop in enumerate(policy) if stop), L-1)
    return k_star, J
```

### 2.3 理論的拡張

#### 2.3.1 リスク調整版（ベイズ的アプローチ）
```python
def bayesian_risk_adjustment(p_hat, n_obs, prior_alpha=1, prior_beta=1):
    """
    Beta事前分布による予測確率の補正
    高分散（n_obs小）の場合は保守的に見積もる
    """
    posterior_alpha = n_obs * p_hat + prior_alpha
    posterior_beta = n_obs * (1 - p_hat) + prior_beta
    return posterior_alpha / (posterior_alpha + posterior_beta)
```

#### 2.3.2 制約付き最適化への変換
```python
def constrained_optimization(tau_max):
    """
    レイテンシ制約下での品質最大化
    max quality s.t. latency ≤ τ_max
    """
    def find_lambda_bisection(low=0.01, high=100.0, tol=1e-3):
        while high - low > tol:
            mid = (low + high) / 2
            latency = measure_average_latency(lambda_val=mid)
            if latency > tau_max:
                low = mid  # λを増やす→早期停止増→レイテンシ減
            else:
                high = mid
        return mid
    
    optimal_lambda = find_lambda_bisection()
    return optimal_lambda
```

## 3. 実装の詳細手順

### 3.1 Stage 0: 環境準備（1日目）

```bash
#!/bin/bash
# setup_environment.sh

# 1. CUDA環境の確認
nvidia-smi
nvcc --version  # CUDA 12.4推奨

# 2. Python環境構築
conda create -n adaptive-sd python=3.10 -y
conda activate adaptive-sd

# 3. 依存関係インストール
pip install torch==2.5.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html
pip install vllm==0.8.3
pip install bitsandbytes==0.45.0
pip install transformers==4.40.0
pip install accelerate==0.35.0

# 4. 評価・開発ツール
pip install numpy scipy pandas matplotlib seaborn
pip install evaluate datasets nltk rouge-score sacrebleu
pip install wandb tensorboard
pip install pytest black flake8 mypy pre-commit

# 5. ディレクトリ構造作成
mkdir -p {src/{models,algorithms,serving,utils},configs,scripts,experiments,tests,data,checkpoints,logs,results}
```

### 3.2 Stage 1: モデル準備（2-3日目）

#### 3.2.1 モデルダウンロードスクリプト
```python
# scripts/download_models.py
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_CONFIGS = {
    "8b": "meta-llama/Llama-3.2-8B",
    "13b": "meta-llama/Llama-3-13B",  # 仮想的な13Bモデル
    "34b": "codellama/CodeLlama-34b-hf",  # 代替として使用
    "70b": "meta-llama/Llama-3.1-70B"
}

def download_and_save(model_size, save_dir="./checkpoints"):
    model_name = MODEL_CONFIGS[model_size]
    print(f"Downloading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    save_path = f"{save_dir}/{model_size}"
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="8b,13b,34b,70b")
    args = parser.parse_args()
    
    for size in args.models.split(","):
        download_and_save(size)
```

#### 3.2.2 量子化実装
```python
# scripts/quantize_models.py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def quantize_model(model_path, output_path, quantization_config):
    """
    4-bit NF4量子化の実装
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model.save_pretrained(output_path)
    print(f"Quantized model saved to {output_path}")
```

### 3.3 Stage 2: コアコンポーネント実装（4-7日目）

#### 3.3.1 Stageクラス
```python
# src/models/stage.py
from typing import List, Tuple, Optional
import torch
from vllm import LLM, SamplingParams
import numpy as np

class Stage:
    def __init__(
        self, 
        model_name: str,
        model_size: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        quantized: bool = True
    ):
        self.model_size = model_size
        self.model_name = model_name
        
        # vLLMエンジン初期化
        self.engine = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization="bitsandbytes" if quantized else None,
            dtype="float16"
        )
        
        # コスト定数（FLOPs/token、実測値で更新）
        self.cost_per_token = {
            "8b": 1.0,
            "13b": 1.6,
            "34b": 4.2,
            "70b": 8.8
        }[model_size]
        
    def generate(
        self, 
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[List[str], np.ndarray]:
        """
        バッチ生成と確率情報の取得
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=5  # top-5 logprobsを取得
        )
        
        outputs = self.engine.generate(prompts, sampling_params)
        
        generated_texts = []
        logprobs_list = []
        
        for output in outputs:
            generated_texts.append(output.outputs[0].text)
            # 各トークンのlogprobsを収集
            token_logprobs = []
            for token_output in output.outputs[0].logprobs:
                token_logprobs.append(token_output)
            logprobs_list.append(token_logprobs)
            
        return generated_texts, np.array(logprobs_list)
    
    def compute_kv_cache_size(self, seq_len: int) -> float:
        """KV-cacheのメモリ使用量推定（GB）"""
        # 簡易計算: 2 * num_layers * hidden_size * seq_len * 2 (K+V) * dtype_size
        size_gb = {
            "8b": seq_len * 0.002,
            "13b": seq_len * 0.003,
            "34b": seq_len * 0.008,
            "70b": seq_len * 0.016
        }[self.model_size]
        return size_gb
```

#### 3.3.2 品質予測モデル
```python
# src/models/predictor.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import time

class QualityPredictor(nn.Module):
    """
    軽量品質予測モデル
    推論時間要件: < 0.3ms @ A100
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        
        # 1層MLP
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 特徴量キャッシュ（高速化）
        self.feature_cache = {}
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)
    
    def predict(
        self, 
        prompt: str,
        draft_output: str,
        draft_logprobs: np.ndarray,
        stage_id: int
    ) -> float:
        """
        合格確率の予測
        """
        start_time = time.time()
        
        # 特徴量抽出
        features = self.feature_extractor.extract(
            prompt, draft_output, draft_logprobs, stage_id
        )
        
        # 推論
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32)
            prob = self.mlp(features_tensor).item()
        
        inference_time = (time.time() - start_time) * 1000
        assert inference_time < 0.3, f"Inference too slow: {inference_time}ms"
        
        return prob

class FeatureExtractor:
    """特徴量抽出器"""
    def extract(
        self,
        prompt: str,
        draft_output: str,
        draft_logprobs: np.ndarray,
        stage_id: int
    ) -> np.ndarray:
        features = []
        
        # 1. 入力特徴
        # エントロピー（最後の32トークン）
        if len(draft_logprobs) > 0:
            entropy = -np.mean([
                np.sum(np.exp(lp) * lp) 
                for lp in draft_logprobs[-32:]
            ])
            features.append(entropy)
        else:
            features.append(0.0)
        
        # 入力長比率
        features.append(len(prompt.split()) / 2048)  # 最大長で正規化
        
        # 2. 出力特徴
        # 生成トークン数
        features.append(len(draft_output.split()) / 512)
        
        # 平均logprob（信頼度の代理指標）
        if len(draft_logprobs) > 0:
            avg_logprob = np.mean([np.max(lp) for lp in draft_logprobs])
            features.append(avg_logprob)
        else:
            features.append(-10.0)
        
        # 3. ステージ特徴
        features.append(stage_id / 4.0)  # 4段階で正規化
        
        # パディング（256次元まで）
        features = np.array(features)
        features = np.pad(features, (0, 256 - len(features)), 'constant')
        
        return features
```

#### 3.3.3 学習データ生成
```python
# scripts/generate_training_data.py
import json
from tqdm import tqdm
import numpy as np

def generate_training_samples(
    stages: List[Stage],
    dataset,
    num_samples: int = 100000
):
    """
    品質予測器の学習データ生成
    """
    training_data = []
    
    for prompt in tqdm(dataset, total=num_samples):
        # 各段を順に実行
        current_prompt = prompt
        
        for i, stage in enumerate(stages[:-1]):  # 最後の70Bは除く
            # Draft生成
            draft_output, draft_logprobs = stage.generate(
                [current_prompt], 
                max_tokens=128
            )
            
            # 70B（ground truth）で検証
            verify_output, _ = stages[-1].generate(
                [current_prompt], 
                max_tokens=128
            )
            
            # 合格判定（簡易版: BLEU > 0.8）
            from sacrebleu import corpus_bleu
            bleu = corpus_bleu(
                [draft_output[0]], 
                [[verify_output[0]]]
            ).score / 100.0
            
            label = 1 if bleu > 0.8 else 0
            
            # 特徴量抽出
            extractor = FeatureExtractor()
            features = extractor.extract(
                current_prompt,
                draft_output[0],
                draft_logprobs[0],
                i
            )
            
            training_data.append({
                "features": features.tolist(),
                "label": label,
                "stage": i,
                "bleu": bleu
            })
            
            # 次段への入力を更新
            current_prompt = current_prompt + " " + draft_output[0]
    
    # 保存
    with open("data/predictor_training.json", "w") as f:
        json.dump(training_data, f)
    
    return training_data
```

### 3.4 Stage 3: パイプライン統合（8-10日目）

#### 3.4.1 動的停止パイプライン
```python
# src/serving/pipeline.py
from typing import List, Dict, Any, Optional
import numpy as np
from src.models.stage import Stage
from src.models.predictor import QualityPredictor
from src.algorithms.dp_solver import optimal_stopping_rule

class AdaptiveSpeculativePipeline:
    def __init__(
        self,
        stages: List[Stage],
        predictor: QualityPredictor,
        lambda_value: float = 1.0,
        risk_adjustment: bool = True
    ):
        self.stages = stages
        self.predictor = predictor
        self.lambda_value = lambda_value
        self.risk_adjustment = risk_adjustment
        
        # 統計情報の追跡
        self.stats = {
            "total_requests": 0,
            "stage_stops": [0] * len(stages),
            "avg_latency": 0,
            "avg_quality": 0
        }
        
    def process_request(
        self,
        prompt: str,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        単一リクエストの処理
        """
        start_time = time.time()
        
        # 各段の予測確率を収集
        probabilities = []
        costs = []
        outputs = []
        
        current_prompt = prompt
        
        for i, stage in enumerate(self.stages):
            # コスト追加
            costs.append(stage.cost_per_token)
            
            # Draft生成
            stage_output, logprobs = stage.generate(
                [current_prompt],
                max_tokens=max_tokens
            )
            outputs.append(stage_output[0])
            
            # 品質予測
            if i < len(self.stages) - 1:  # 最終段以外
                p_accept = self.predictor.predict(
                    current_prompt,
                    stage_output[0],
                    logprobs[0],
                    i
                )
                
                # リスク調整
                if self.risk_adjustment:
                    n_obs = len(self.stats["stage_stops"])
                    p_accept = self._bayesian_adjustment(p_accept, n_obs)
                
                probabilities.append(p_accept)
            else:
                probabilities.append(1.0)  # 最終段は常に1.0
            
            # 最適停止判定
            k_star, _ = optimal_stopping_rule(
                probabilities[:i+1],
                costs[:i+1],
                self.lambda_value
            )
            
            if k_star == i:  # 現段で停止
                break
                
            # 次段への入力準備
            current_prompt = current_prompt + " " + stage_output[0]
        
        # 統計更新
        self._update_stats(k_star, time.time() - start_time)
        
        return {
            "output": outputs[k_star],
            "stopped_at_stage": k_star,
            "latency_ms": (time.time() - start_time) * 1000,
            "probabilities": probabilities,
            "costs": costs[:k_star+1]
        }
    
    def _bayesian_adjustment(
        self,
        p_hat: float,
        n_obs: int,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> float:
        """ベイズ的リスク調整"""
        return (n_obs * p_hat + alpha) / (n_obs + alpha + beta)
    
    def _update_stats(self, stage: int, latency: float):
        """統計情報の更新"""
        self.stats["total_requests"] += 1
        self.stats["stage_stops"][stage] += 1
        
        # 移動平均
        alpha = 0.01
        self.stats["avg_latency"] = (
            (1 - alpha) * self.stats["avg_latency"] + 
            alpha * latency * 1000
        )
```

#### 3.4.2 KV-Cacheマネージャー
```python
# src/serving/cache_manager.py
import torch
from typing import Dict, List, Optional

class KVCacheManager:
    """
    動的停止に対応したKV-Cache管理
    """
    def __init__(self, num_stages: int):
        self.num_stages = num_stages
        self.caches: Dict[int, Dict] = {}
        self.memory_usage = {}
        
    def allocate(self, request_id: str, stage_id: int, kv_cache: Dict):
        """特定段のKV-Cacheを保存"""
        if request_id not in self.caches:
            self.caches[request_id] = {}
        
        self.caches[request_id][stage_id] = kv_cache
        self.memory_usage[request_id] = self._calculate_memory(kv_cache)
        
    def truncate_at_stage(self, request_id: str, final_stage: int):
        """
        停止段より後のKV-Cacheを解放
        """
        if request_id in self.caches:
            stages_to_delete = [
                s for s in self.caches[request_id].keys() 
                if s > final_stage
            ]
            
            for stage in stages_to_delete:
                del self.caches[request_id][stage]
                
            # GPUメモリ解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def get_cache(self, request_id: str, stage_id: int) -> Optional[Dict]:
        """特定段のKV-Cache取得"""
        if request_id in self.caches:
            return self.caches[request_id].get(stage_id)
        return None
    
    def cleanup(self, request_id: str):
        """リクエスト完了後のクリーンアップ"""
        if request_id in self.caches:
            del self.caches[request_id]
            del self.memory_usage[request_id]
            torch.cuda.empty_cache()
            
    def _calculate_memory(self, kv_cache: Dict) -> float:
        """KV-Cacheのメモリ使用量計算（GB）"""
        total_bytes = 0
        for k, v in kv_cache.items():
            if isinstance(v, torch.Tensor):
                total_bytes += v.element_size() * v.nelement()
        return total_bytes / (1024 ** 3)
```

### 3.5 Stage 4: サービング実装（11-12日目）

#### 3.5.1 FastAPIサーバ
```python
# src/serving/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import List, Optional
import logging

app = FastAPI(title="Adaptive Speculative Decoding API")

# グローバル変数
pipeline = None
cache_manager = None

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    output: str
    stopped_at_stage: int
    latency_ms: float
    stage_probabilities: List[float]

@app.on_event("startup")
async def startup_event():
    """サーバ起動時の初期化"""
    global pipeline, cache_manager
    
    # モデルロード
    stages = []
    for size in ["8b", "13b", "34b", "70b"]:
        stage = Stage(
            model_name=f"./checkpoints/{size}",
            model_size=size,
            tensor_parallel_size=get_tp_size(size)
        )
        stages.append(stage)
    
    # 予測器ロード
    predictor = QualityPredictor()
    predictor.load_state_dict(
        torch.load("./checkpoints/predictor.pt")
    )
    
    # パイプライン初期化
    pipeline = AdaptiveSpeculativePipeline(
        stages=stages,
        predictor=predictor,
        lambda_value=float(os.getenv("LAMBDA_VALUE", "1.0"))
    )
    
    # キャッシュマネージャー
    cache_manager = KVCacheManager(num_stages=4)
    
    logging.info("Server initialized successfully")

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """生成エンドポイント"""
    try:
        # 非同期実行
        result = await asyncio.to_thread(
            pipeline.process_request,
            request.prompt,
            request.max_tokens
        )
        
        return GenerationResponse(
            output=result["output"],
            stopped_at_stage=result["stopped_at_stage"],
            latency_ms=result["latency_ms"],
            stage_probabilities=result["probabilities"]
        )
        
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """統計情報エンドポイント"""
    return pipeline.stats

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy"}

def get_tp_size(model_size: str) -> int:
    """モデルサイズに応じたTensor Parallel数"""
    return {
        "8b": 1,
        "13b": 2,
        "34b": 2,
        "70b": 4
    }[model_size]

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # GPU共有のため単一ワーカー
        log_level="info"
    )
```

### 3.6 Stage 5: 評価実装（13-15日目）

#### 3.6.1 評価スクリプト
```python
# experiments/evaluate_pipeline.py
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from evaluate import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PipelineEvaluator:
    def __init__(self, pipeline, datasets_config):
        self.pipeline = pipeline
        self.datasets = self._load_datasets(datasets_config)
        
        # 評価メトリクス
        self.bleu = load("bleu")
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        
    def evaluate_all(self, num_samples=1000):
        """全データセットでの評価"""
        results = {}
        
        for dataset_name, dataset in self.datasets.items():
            print(f"\nEvaluating on {dataset_name}...")
            
            dataset_results = {
                "latencies": [],
                "qualities": [],
                "stage_distribution": [0] * len(self.pipeline.stages),
                "cost_savings": []
            }
            
            # サンプル評価
            for sample in tqdm(dataset[:num_samples]):
                result = self._evaluate_sample(sample)
                
                dataset_results["latencies"].append(result["latency"])
                dataset_results["qualities"].append(result["quality"])
                dataset_results["stage_distribution"][result["stage"]] += 1
                dataset_results["cost_savings"].append(result["cost_saving"])
            
            # 統計計算
            dataset_results["stats"] = {
                "latency_p50": np.percentile(dataset_results["latencies"], 50),
                "latency_p95": np.percentile(dataset_results["latencies"], 95),
                "latency_p99": np.percentile(dataset_results["latencies"], 99),
                "avg_quality": np.mean(dataset_results["qualities"]),
                "avg_cost_saving": np.mean(dataset_results["cost_savings"]),
                "stage_distribution": dataset_results["stage_distribution"]
            }
            
            results[dataset_name] = dataset_results
            
        return results
    
    def _evaluate_sample(self, sample):
        """単一サンプルの評価"""
        prompt = sample["prompt"]
        reference = sample.get("reference", "")
        
        # パイプライン実行
        output = self.pipeline.process_request(prompt)
        
        # 品質評価（referenceがある場合）
        quality = 1.0  # デフォルト
        if reference:
            quality = self._calculate_quality(
                output["output"], 
                reference
            )
        
        # コスト計算
        actual_cost = sum(output["costs"])
        max_cost = sum([s.cost_per_token for s in self.pipeline.stages])
        cost_saving = 1.0 - (actual_cost / max_cost)
        
        return {
            "latency": output["latency_ms"],
            "quality": quality,
            "stage": output["stopped_at_stage"],
            "cost_saving": cost_saving
        }
    
    def _calculate_quality(self, generated: str, reference: str):
        """品質スコア計算（BLEU, ROUGE, BERTScoreの組み合わせ）"""
        # BLEU
        bleu_score = self.bleu.compute(
            predictions=[generated],
            references=[[reference]]
        )["bleu"]
        
        # ROUGE-L
        rouge_score = self.rouge.compute(
            predictions=[generated],
            references=[reference]
        )["rougeL"]
        
        # 重み付き平均
        return 0.5 * bleu_score + 0.5 * rouge_score
    
    def visualize_results(self, results, output_dir="./results"):
        """結果の可視化"""
        # 1. レイテンシ分布
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for idx, (dataset_name, data) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            ax.hist(data["latencies"], bins=50, alpha=0.7)
            ax.axvline(data["stats"]["latency_p50"], color='r', 
                      linestyle='--', label='p50')
            ax.axvline(data["stats"]["latency_p95"], color='g', 
                      linestyle='--', label='p95')
            ax.set_title(f"{dataset_name} - Latency Distribution")
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Count")
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_distribution.png")
        
        # 2. ステージ分布
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stage_data = pd.DataFrame({
            dataset: data["stats"]["stage_distribution"]
            for dataset, data in results.items()
        })
        
        stage_data.plot(kind='bar', ax=ax)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Number of Requests")
        ax.set_title("Stage Distribution by Dataset")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/stage_distribution.png")
        
        # 3. 品質-レイテンシトレードオフ
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dataset_name, data in results.items():
            ax.scatter(
                data["stats"]["latency_p50"],
                data["stats"]["avg_quality"],
                s=100,
                label=dataset_name
            )
        
        ax.set_xlabel("Median Latency (ms)")
        ax.set_ylabel("Average Quality")
        ax.set_title("Quality-Latency Tradeoff")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/quality_latency_tradeoff.png")
    
    def _load_datasets(self, config):
        """データセットのロード"""
        datasets = {}
        
        # MMLU
        if "mmlu" in config:
            mmlu = load_dataset("cais/mmlu", "all")["test"]
            datasets["mmlu"] = [
                {"prompt": f"Question: {q}\nAnswer:", 
                 "reference": a}
                for q, a in zip(mmlu["question"], mmlu["answer"])
            ]
        
        # HumanEval
        if "humaneval" in config:
            humaneval = load_dataset("openai_humaneval")["test"]
            datasets["humaneval"] = [
                {"prompt": sample["prompt"]}
                for sample in humaneval
            ]
        
        # 他のデータセットも同様に追加
        
        return datasets

# 実行
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", type=float, default=1.0)
    parser.add_argument("--datasets", default="mmlu,humaneval")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="./results")
    
    args = parser.parse_args()
    
    # パイプライン初期化
    pipeline = load_pipeline(lambda_value=args.lambda)
    
    # 評価実行
    evaluator = PipelineEvaluator(
        pipeline,
        args.datasets.split(",")
    )
    
    results = evaluator.evaluate_all(args.num_samples)
    
    # 結果保存
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 可視化
    evaluator.visualize_results(results, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")
```

### 3.7 Stage 6: ベースライン比較（16-17日目）

#### 3.7.1 ベースライン実装
```python
# experiments/baselines.py

class FixedDepthBaseline:
    """固定深度Speculative Decoding"""
    def __init__(self, draft_model, verify_model):
        self.draft = draft_model
        self.verify = verify_model
        
    def generate(self, prompt, max_tokens=512):
        # 常に2段構成
        draft_output = self.draft.generate(prompt, max_tokens)
        verify_output = self.verify.generate(prompt, max_tokens)
        return verify_output

class PipeSpecBaseline:
    """PipeSpec実装（簡易版）"""
    def __init__(self, models, k=4):
        self.models = models
        self.k = k  # 固定段数
        
    def generate(self, prompt, max_tokens=512):
        # k段を非同期実行（簡易実装）
        outputs = []
        for i in range(self.k):
            output = self.models[i].generate(prompt, max_tokens)
            outputs.append(output)
        return outputs[-1]  # 最終段を返す

# 比較実験
def compare_methods(test_data, methods_dict):
    """各手法の比較"""
    results = {}
    
    for method_name, method in methods_dict.items():
        print(f"Evaluating {method_name}...")
        
        latencies = []
        qualities = []
        
        for sample in tqdm(test_data):
            start = time.time()
            output = method.generate(sample["prompt"])
            latency = (time.time() - start) * 1000
            
            quality = calculate_quality(output, sample["reference"])
            
            latencies.append(latency)
            qualities.append(quality)
        
        results[method_name] = {
            "avg_latency": np.mean(latencies),
            "avg_quality": np.mean(qualities),
            "latency_std": np.std(latencies),
            "quality_std": np.std(qualities)
        }
    
    return results
```

## 4. 実験プロトコル

### 4.1 小規模実験（PoC）
```bash
# 2段構成での基礎検証
python experiments/poc_2stage.py \
    --draft-model llama-8b \
    --verify-model llama-13b \
    --test-samples 100 \
    --lambda-values "0.1,0.5,1.0,2.0"
```

### 4.2 フルスケール実験
```bash
# 全4段での本実験
bash experiments/run_full_evaluation.sh
```

内容：
```bash
#!/bin/bash
# experiments/run_full_evaluation.sh

# パラメータグリッド
LAMBDAS="0.1 0.2 0.5 1.0 2.0 5.0"
DATASETS="mmlu humaneval hotpotqa alpacaeval longbench"

for lambda in $LAMBDAS; do
    for dataset in $DATASETS; do
        echo "Running lambda=$lambda, dataset=$dataset"
        
        python experiments/evaluate_pipeline.py \
            --lambda $lambda \
            --datasets $dataset \
            --num-samples 1000 \
            --output-dir results/lambda_${lambda}/
            
        # GPUクールダウン
        sleep 30
    done
done

# 結果集計
python experiments/aggregate_results.py
```

### 4.3 アブレーション実験

```python
# experiments/ablation_studies.py

def ablation_predictor_features():
    """予測器の特徴量アブレーション"""
    feature_sets = {
        "full": ["entropy", "length", "logprobs", "stage"],
        "no_entropy": ["length", "logprobs", "stage"],
        "no_logprobs": ["entropy", "length", "stage"],
        "minimal": ["length", "stage"]
    }
    
    for name, features in feature_sets.items():
        # 各特徴量セットで学習・評価
        pass

def ablation_risk_adjustment():
    """リスク調整の効果検証"""
    configs = [
        {"risk_adjust": False},
        {"risk_adjust": True, "alpha": 1, "beta": 1},
        {"risk_adjust": True, "alpha": 2, "beta": 2}
    ]
    
    for config in configs:
        # 各設定で評価
        pass
```

## 5. トラブルシューティング

### 5.1 よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| OOM (Out of Memory) | 70Bモデルが大きすぎる | 1. 量子化レベルを上げる（8bit→4bit）<br>2. gradient_checkpointingを有効化<br>3. バッチサイズを1に |
| 予測器の精度が低い | 学習データ不足/偏り | 1. データ拡張（paraphrase）<br>2. 難易度別にバランシング<br>3. アンサンブル学習 |
| レイテンシが期待より高い | パイプライン overhead | 1. 特徴量計算をGPU化<br>2. 予測器をONNXに変換<br>3. vLLMのcontinuous batchingを調整 |

### 5.2 デバッグ用ツール

```python
# src/utils/debug.py

class PipelineDebugger:
    def trace_request(self, prompt):
        """リクエストの詳細トレース"""
        print(f"=== Processing: {prompt[:50]}... ===")
        
        for i, stage in enumerate(self.pipeline.stages):
            print(f"\n--- Stage {i} ({stage.model_size}) ---")
            
            output, logprobs = stage.generate([prompt])
            prob = self.pipeline.predictor.predict(...)
            
            print(f"Output: {output[0][:100]}...")
            print(f"Predicted acceptance: {prob:.3f}")
            print(f"Should stop: {self._should_stop(i, prob)}")
            
def profile_inference():
    """推論のプロファイリング"""
    import torch.profiler
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # 推論実行
        pipeline.process_request("Test prompt")
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 6. 期待される結果と成功基準

### 6.1 定量的目標
- レイテンシ削減: 固定2段比で 30-50% 削減
- 品質維持: BLEU/ROUGEの劣化 < 2%
- コスト削減: GPU使用時間 40% 削減

### 6.2 定性的目標
- 簡単な質問は8B/13Bで早期停止
- 複雑な質問のみ70Bまで到達
- λ調整による柔軟な品質-速度制御

## 7. 研究の拡張可能性

1. **モデル非依存化**: Llama以外への適用
2. **分散実装**: 複数ノードでの並列化
3. **オンライン学習**: 予測器の継続的改善
4. **バッチ最適化**: 難易度別動的バッチング

## まとめ

本プロトコルは、Adaptive Speculative Decodingの完全な実装手順を提供しています。各ステージを順に実装し、評価を行うことで、入力依存の動的深度最適化という新しいパラダイムを実現できます。