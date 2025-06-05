# Getting Started with Adaptive Speculative Decoding

## 研究開始の手順

### 1. 初期セットアップ（Day 1）

```bash
# リポジトリのクローン後
cd adaptive-speculative-decoding

# 環境構築
bash scripts/setup_env.sh
conda activate adaptive-sd
pip install -r requirements.txt
```

### 2. 最小構成での動作確認（Day 2-3）

まずは2段構成（8B→13B）で基本的な動作を確認：

```python
# src/models/simple_test.py として作成
from src.models.stage import Stage

# 8Bモデルのみでテスト
stage_8b = Stage(
    model_name="meta-llama/Llama-3.2-8B",
    model_size="8b",
    tensor_parallel_size=1
)

# 簡単なプロンプトでテスト
output, logprobs = stage_8b.generate(
    ["What is 2+2?"],
    max_tokens=10
)
print(f"Output: {output[0]}")
```

### 3. 品質予測器の実装とテスト（Day 4-5）

```python
# 簡易版の品質予測器でテスト
import numpy as np
from src.models.predictor import QualityPredictor

# ダミーデータで動作確認
predictor = QualityPredictor(feature_dim=256)
dummy_features = np.random.randn(256)
prob = predictor(torch.tensor(dummy_features, dtype=torch.float32))
print(f"Predicted probability: {prob.item()}")
```

### 4. 動的計画法の検証（Day 6）

```python
# アルゴリズムの単体テスト
from src.algorithms.dp_solver import optimal_stopping_rule

# テストケース
p = [0.3, 0.5, 0.7, 0.9]  # 各段の合格確率
C = [1.0, 1.6, 4.2, 8.8]  # 各段のコスト
lam = 1.0  # 品質重み

k_star, J = optimal_stopping_rule(p, C, lam)
print(f"Optimal stopping stage: {k_star}")
print(f"Expected costs: {J}")
```

### 5. 2段パイプラインのPoC（Day 7-8）

```bash
# experiments/poc_2stage.py を実行
python experiments/poc_2stage.py \
    --model1 llama-3-8b \
    --model2 llama-3-13b \
    --dataset mmlu-subset \
    --num-samples 10 \
    --lambda 1.0
```

### 6. 評価と可視化（Day 9-10）

```python
# 結果の可視化
import matplotlib.pyplot as plt
import json

# 結果読み込み
with open("results/poc_results.json", "r") as f:
    results = json.load(f)

# レイテンシ vs 品質のプロット
plt.scatter(results["latencies"], results["qualities"])
plt.xlabel("Latency (ms)")
plt.ylabel("Quality Score")
plt.title("Quality-Latency Tradeoff")
plt.savefig("results/tradeoff.png")
```

## 主要な実装チェックポイント

### ✅ Stage 1: 基礎実装
- [ ] 単一モデルの動作確認
- [ ] vLLMの基本的な使い方の理解
- [ ] 量子化の動作確認

### ✅ Stage 2: 品質予測器
- [ ] 特徴量抽出の実装
- [ ] MLPモデルの実装
- [ ] 推論時間 < 0.3ms の確認

### ✅ Stage 3: 最適停止則
- [ ] 動的計画法の実装
- [ ] λパラメータの影響確認
- [ ] 計算量O(L)の検証

### ✅ Stage 4: パイプライン統合
- [ ] 2段での動作確認
- [ ] KV-Cache管理の実装
- [ ] 早期停止の動作確認

### ✅ Stage 5: 評価システム
- [ ] メトリクスの実装
- [ ] ベースラインとの比較
- [ ] 結果の可視化

## トラブルシューティング

### GPU関連
```bash
# CUDAが使えるか確認
python -c "import torch; print(torch.cuda.is_available())"

# vLLMのGPU設定確認
python -c "from vllm import LLM; print(LLM.get_supported_archs())"
```

### メモリ不足
```python
# 量子化レベルを上げる
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True  # より積極的な量子化
)
```

### 品質予測器の精度が低い
```python
# 特徴量の重要度を確認
from sklearn.inspection import permutation_importance

importance = permutation_importance(
    predictor, X_test, y_test, n_repeats=10
)
print("Feature importance:", importance.importances_mean)
```

## 次のステップ

1. **小規模実験の完了**: 2段構成で基本的な動作を確認
2. **λパラメータの調整**: 品質-速度トレードオフの最適化
3. **4段への拡張**: 全モデルを使った完全なパイプライン
4. **大規模評価**: 全データセットでの性能測定
5. **論文執筆**: 結果をまとめて投稿準備

## 研究ノート

実験を進める際は、以下の情報を記録してください：

```markdown
## 実験ログ (YYYY-MM-DD)

### 実験設定
- モデル構成: 
- λ値: 
- データセット: 
- サンプル数: 

### 結果
- 平均レイテンシ: 
- 平均品質: 
- 早期停止率: 

### 観察
- 
- 

### 次の実験
- 
```

頑張ってください！