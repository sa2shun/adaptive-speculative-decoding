# Adaptive Speculative Decoding

多段Draft-Verifyパイプラインによる入力依存型推論深度最適化フレームワーク

## 研究概要

大規模言語モデル（LLM）の推論コストを入力難易度に応じて動的に最適化する手法。8B→13B→34B→70Bの階層化されたモデルを用い、各問い合わせごとに「どの段で推論を打ち切るか」をオンラインで決定する。

### 主要な貢献

1. **多段Draft-Verify体系化**: 従来の固定2段構成を拡張し、入力依存で停止深度を最適化
2. **軽量品質予測モデル**: 数ms以内に段ごとの合格確率を推定
3. **動的計画法による最適停止則**: 期待コストと期待品質損失の線形結合を最小化

## 既存研究との差別化

| 手法 | パイプライン深度 | 深度の動的調整 | 本研究との違い |
|------|-----------------|----------------|--------------|
| PipeSpec | 2〜k段（固定） | ✗ | 深度は静的設定、本研究は動的最適化 |
| BanditSpec | 2段固定 | ✗（ハイパラのみ適応） | モデル階層自体は固定、本研究は階層選択を最適化 |
| ML-SpecQD | 多段（量子化） | ✗ | 最終段verify必須、本研究は途中段で停止可能 |
| DEL | 単一モデル | ✓（層内） | 単一モデル内部の層で終了、本研究はモデル間階層 |

## システム構成

```
┌──────────────┐
│ REST / gRPC  │  ←── クライアント
└──────┬───────┘
       │ prompt
┌──────▼─────────┐
│ Stage-0: 8B    │ draft₁(y₁…yₙ)
└──┬─────────────┘
   │ accept_prob p̂₁
┌──▼─────────────┐
│Stop Checker    │→ Yes → return to user
│                │
│No              │
│                ▼
│          Stage-1: 13B
│                │
│               ...
│                │
│          Stage-3: 70B
└────────────────┘
```

## 理論的基礎

### 最適化問題

入力xに対して、停止段k(x)を以下の目的関数で決定：

```
min_{k(x)} E[Σᵢ₌₁^{k(x)} cᵢ] + λ E[1 - p̄_{k(x)}]
```

- cᵢ: 段iの1トークン当たりコスト
- p̄ᵢ: 段iまでの累積合格確率（Πⱼ≤ᵢ pⱼ）
- λ: 品質重み（品質-速度トレードオフパラメータ）

### 動的計画法アルゴリズム

```python
def optimal_cut(p, C, lam):
    """
    p: 各段の合格確率リスト
    C: 各段のコストリスト
    lam: 品質重みパラメータ
    """
    L = len(C)
    J = [0] * (L + 1)
    stop = [False] * L
    
    for i in reversed(range(L)):
        cost_if_stop = C[i] + lam * (1 - np.prod(p[:i+1]))
        cost_if_go = C[i] + J[i+1]
        stop[i] = cost_if_stop <= cost_if_go
        J[i] = min(cost_if_stop, cost_if_go)
    
    k_star = next((i for i, s in enumerate(stop) if s), L-1)
    return k_star
```

## 実装詳細

### 環境構築

```bash
# CUDA 12.4 / cuDNN 9を想定
conda create -n adaptive-sd python=3.10 -y
conda activate adaptive-sd

# コア実行エンジン
pip install "vllm>=0.8.3"         # 推論サーバ
pip install "bitsandbytes>=0.45"  # 4/8-bit量子化
pip install "transformers>=4.40"  # モデルローダー

# 解析・評価
pip install numpy scipy tqdm evaluate wandb datasets
pip install rouge-score sacrebleu nltk

# 開発ツール
pip install pytest black flake8 mypy
```

### モデル準備

```bash
# Llama-3系列の取得と量子化
python scripts/download_models.py --models 8b,13b,34b,70b
python scripts/quantize_models.py --method nf4 --models all
```

### データセット

| タスク | 用途 | 難易度の多様性確保 |
|--------|------|-------------------|
| MMLU | 汎用QA | 教科カテゴリごとの難易度差 |
| HumanEval | コード生成 | prompt長さで3分位分割 |
| HotpotQA | 事実質問 | bridge vs single-hop |
| AlpacaEval2 | 会話体 | 雑談の難易度バリエーション |
| LongBench | 長文生成 | 文脈長による複雑度変化 |

## 品質予測モデル

### 特徴量設計

1. **入力特徴**
   - prompt末尾32トークンのエントロピー
   - 入力長の比率（現在長/最大長）
   - question-typeエンベディング

2. **draft-verify間特徴**
   - KL(logits_draft, logits_upper)
   - acceptance rate履歴（過去100トークン）

3. **モデルアーキテクチャ**
   - 1層MLP: 256 → 128 → 1
   - 活性化: ReLU → Sigmoid
   - 推論時間要件: < 0.3ms @ A100

### 学習方法

```python
# データ生成
for prompt in dataset:
    outputs_8b = model_8b.generate(prompt)
    outputs_70b = model_70b.generate(prompt)
    label = calculate_acceptance(outputs_8b, outputs_70b)
    features = extract_features(prompt, outputs_8b)
    training_data.append((features, label))

# 学習（クラス不均衡対応）
class_weights = compute_class_weight(training_labels)
model = train_predictor(training_data, class_weights)
```

## 実装上の課題と解決策

### 1. GPU配置とメモリ管理

**課題**: 70Bモデルのメモリ占有が大きい

**解決策**:
```python
# Dynamic GPU Pooling
torch.cuda.set_enabled_lms(True)  # Large Model Support
vllm_config = {
    "gpu_memory_utilization": 0.8,
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 4096
}
```

### 2. KV-Cache整合性

**課題**: 途中停止時の上位段KV欠損

**解決策**:
```python
class KVCacheManager:
    def truncate_at_stage(self, stage_id):
        # 停止段のKVのみ保持
        for i in range(stage_id + 1, self.num_stages):
            del self.caches[i]
        torch.cuda.empty_cache()
```

### 3. 品質予測器の精度

**課題**: 予測誤差による品質劣化

**解決策**: ベイズ的リスク調整
```python
def risk_adjusted_prediction(p_hat, n_samples, alpha=1, beta=1):
    # Beta事前分布によるshrinkage
    return (n_samples * p_hat + alpha) / (n_samples + alpha + beta)
```

### 4. λパラメータ調整

**課題**: サービスSLOに依存した手動調整

**解決策**: 制約付き最適化への変換
```python
# latency ≤ τ制約下での品質最大化
def find_optimal_lambda(tau_ms):
    return binary_search_lambda(
        constraint=lambda l: measure_latency(l) <= tau_ms,
        objective=lambda l: measure_quality(l)
    )
```

## 評価プロトコル

### メトリクス

1. **性能指標**
   - レイテンシ: p50, p95, p99
   - スループット: requests/sec
   - GPU使用率・エネルギー効率

2. **品質指標**
   - BLEU, ROUGE-L
   - BERTScore
   - 人手評価（fluency, coherence, correctness）

3. **システム指標**
   - 各段acceptance rate
   - 予測器calibration (ECE, AUC)
   - 早期停止率分布

### ベースライン比較

```bash
# 固定2段SD
python experiments/run_baseline.py --method fixed-2stage

# PipeSpec (k=4, async)
python experiments/run_baseline.py --method pipespec --stages 4

# BanditSpec (UCB)
python experiments/run_baseline.py --method banditspec

# DEL (self-exit)
python experiments/run_baseline.py --method del
```

## プロジェクト構造

```
adaptive-speculative-decoding/
├── README.md              # 本ファイル
├── requirements.txt       # 依存関係
├── setup.py              # パッケージ設定
├── configs/              # 設定ファイル
│   ├── models.yaml       # モデル設定
│   ├── training.yaml     # 学習設定
│   └── serving.yaml      # サービング設定
├── src/                  # ソースコード
│   ├── models/           # モデル実装
│   │   ├── stage.py      # Stageクラス
│   │   └── predictor.py  # 品質予測器
│   ├── algorithms/       # アルゴリズム
│   │   ├── dp_solver.py  # 動的計画法
│   │   └── optimizer.py  # λ最適化
│   ├── serving/          # サービング
│   │   ├── server.py     # FastAPIサーバ
│   │   └── pipeline.py   # パイプライン制御
│   └── utils/            # ユーティリティ
├── scripts/              # 実行スクリプト
│   ├── download_models.py
│   ├── train_predictor.py
│   └── evaluate.py
├── experiments/          # 実験スクリプト
│   ├── run_mmlu.sh
│   ├── run_humaneval.sh
│   └── ablation/         # アブレーション実験
├── tests/                # テストコード
├── data/                 # データセット
├── checkpoints/          # モデルチェックポイント
├── logs/                 # ログファイル
└── results/              # 実験結果
```

## クイックスタート

```bash
# 1. 環境構築
bash scripts/setup_env.sh

# 2. モデルダウンロード
python scripts/download_models.py

# 3. 品質予測器の学習
python scripts/train_predictor.py --config configs/training.yaml

# 4. サーバ起動
python -m src.serving.server --config configs/serving.yaml

# 5. 評価実行
bash experiments/run_all.sh
```

## 実験の再現

### 最小構成での動作確認

```bash
# 2段構成（8B→13B）でのPoC
python experiments/poc_2stage.py \
    --model1 llama-3-8b \
    --model2 llama-3-13b \
    --dataset mmlu-subset \
    --lambda 0.5
```

### フル実験の実行

```bash
# 全ベンチマーク・全設定での評価
python experiments/full_evaluation.py \
    --stages 4 \
    --models 8b,13b,34b,70b \
    --datasets all \
    --lambda-range 0.1:2.0:0.1 \
    --output-dir results/$(date +%Y%m%d)
```

## 今後の拡張

1. **モデル非依存化**: Llama以外のモデルファミリーへの対応
2. **動的バッチング**: 難易度別バッチ形成による効率化
3. **分散推論**: 複数ノードでのパイプライン実行
4. **オンライン学習**: 品質予測器の継続的改善

## ライセンス

Apache License 2.0

## 引用

```bibtex
@article{adaptive-speculative-decoding,
  title={Adaptive Speculative Decoding with Multi-Stage Draft-Verify Pipeline},
  author={Your Name},
  year={2025}
}
```