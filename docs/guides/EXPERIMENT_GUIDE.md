# 実験実行ガイド

## 🚀 クイックスタート

H100が8枚ある環境で、以下のコマンドを実行するだけで全実験が完了します：

```bash
./run_complete_experiments.sh
```

## 📋 実験の流れ

### 1. **事前準備** (自動)
- GPUの確認
- 依存関係のインストール
- ディレクトリ作成

### 2. **モデルダウンロード** (2-3時間)
```bash
# 個別実行する場合
python3 scripts/download_qwen2.5_models.py --base-path /raid/$USER/models
```
- Qwen2.5-7B (15GB)
- Qwen2.5-14B (28GB)
- Qwen2.5-32B (65GB)
- Qwen2.5-72B (145GB)

### 3. **データセット準備** (30分)
```bash
# 個別実行する場合
python3 setup_datasets.py --output-dir /raid/$USER/datasets
```
- MMLU: 2000サンプル
- HumanEval: 164サンプル
- MT-Bench風: 100サンプル
- Simple Q&A: 500サンプル

### 4. **訓練データ生成** (4-5時間)
```bash
# 個別実行する場合
python3 src/training/generate_training_data.py \
    --dataset /raid/$USER/datasets/mmlu_test.json \
    --model-paths /raid/$USER/models/model_paths.json \
    --output-dir /raid/$USER/training_data
```
- 各モデルで推論実行
- BLEUスコア計算
- 特徴量抽出

### 5. **品質予測器の学習** (1時間)
- 32次元MLPの学習
- 50エポック
- 検証精度の確認

### 6. **評価実験** (2時間)
- 理論的検証
- 実データでの評価
- 統計的検定

## 📊 出力ファイル

実験完了後、以下のファイルが生成されます：

```
/raid/$USER/adaptive-speculative-decoding/results/run_YYYYMMDD_HHMMSS/
├── experiment_report.md      # 実験レポート
├── evaluation_results.json   # 評価結果
├── theoretical_results_simple.png  # 理論的結果の図
├── paper_results.png        # 論文用の図
└── paper_results.pdf        # 論文用の図（PDF）
```

## 🔍 ログの確認

すべてのログは以下に保存されます：
```
/raid/$USER/adaptive-speculative-decoding/logs/
├── experiment_YYYYMMDD_HHMMSS.log      # メインログ
├── model_download_YYYYMMDD_HHMMSS.log  # モデルダウンロードログ
├── dataset_setup_YYYYMMDD_HHMMSS.log   # データセット準備ログ
├── training_data_gen_YYYYMMDD_HHMMSS.log  # 訓練データ生成ログ
├── predictor_training_YYYYMMDD_HHMMSS.log # 予測器学習ログ
└── real_experiments_YYYYMMDD_HHMMSS.log   # 評価実験ログ
```

## ⚡ GPU使用状況

実験中のGPU割り当て：
- GPU 0: Qwen2.5-7B
- GPU 1: Qwen2.5-14B
- GPU 2-3: Qwen2.5-32B
- GPU 4-7: Qwen2.5-72B

## 🛠️ トラブルシューティング

### モデルダウンロードが失敗する場合
```bash
# HuggingFaceトークンを設定
export HF_TOKEN="your_token_here"

# 再度ダウンロード
python3 scripts/download_qwen2.5_models.py --token $HF_TOKEN
```

### メモリ不足の場合
```bash
# バッチサイズを小さくして再実行
python3 src/training/generate_training_data.py \
    --batch-size 4  # デフォルトは8
```

### 実験を途中から再開する場合
```bash
# 既存のモデル・データを使って実験のみ実行
python3 run_real_experiments.py --skip-training
```

## 📈 期待される結果

正常に完了した場合：
- **平均高速化**: 3-4倍
- **コスト削減**: 60-70%
- **品質保持**: 95%以上
- **統計的有意性**: p < 0.001

## 💡 Tips

1. **ディスク容量**: 最低300GB必要
2. **実行時間**: 全体で約10時間
3. **並列実行**: モデルダウンロードは並列化されています
4. **中断対応**: 各フェーズは独立しているため、途中から再開可能

## 🎯 次のステップ

実験完了後：
1. `experiment_report.md`で結果を確認
2. `PAPER.md`の数値を実測値で更新
3. 図表を論文に組み込み
4. 追加実験が必要な場合は個別スクリプトを実行