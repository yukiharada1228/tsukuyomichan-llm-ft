# つくよみちゃん LLM ファインチューニング

[つくよみちゃん会話AI育成計画](https://tyc.rei-yumesaki.net/material/kaiwa-ai/)の公式データを使用して、日本語LLMをSFT（Supervised Fine-Tuning）でファインチューニングするプロジェクトです。

## クイックスタート

```bash
ollama run hf.co/yukiharada1228/tsukuyomichan-sarashina2.2-3b-instruct-v0.1-GGUF:Q4_K_M
```

## ✨ 特徴

- **SFT（Supervised Fine-Tuning）** によるファインチューニング
- **QLoRA + LoRA** によるメモリ効率的な学習
- **シンプルなPythonスクリプト** による学習・マージワークフロー
- **llama.cpp** によるGGUF形式への変換と効率的な実行

## 🚀 クイックスタート

### 1. リポジトリのクローン

```bash
git clone --recursive https://github.com/yukiharada1228/tsukuyomichan-llm-ft.git
cd tsukuyomichan-llm-ft
```

### 2. 環境のセットアップ

このプロジェクトは `uv` を使用して依存関係を管理します。

```bash
# 依存関係のインストール
uv sync
```

### 3. 学習データの準備

```bash
# データのダウンロードと変換
uv run download_and_convert.py
```

これにより、`./data/tsukuyomi.jsonl` が生成されます。

### 4. 学習の実行

```bash
uv run sft_train.py
```

学習結果は `./results/{model_path}/sft_model/` に保存されます。

### 5. モデルのマージ（オプション）

SFTアダプターをベースモデルにマージして統合モデルを作成できます。

```bash
uv run merge_model.py
```

マージされたモデルは `./results/{base_model}/merged_model/` に保存されます。

## 📁 プロジェクト構成

```
tsukuyomichan-llm-ft/
├── download_and_convert.py    # データダウンロード・変換スクリプト
├── sft_train.py                # SFT学習スクリプト
├── merge_model.py              # モデルマージスクリプト
├── main.py                     # エントリーポイント（予定）
├── pyproject.toml              # プロジェクト設定と依存関係
├── uv.lock                     # 依存関係ロックファイル
├── data/                       # 学習データ
│   └── tsukuyomi.jsonl         # 変換済み学習データ
├── results/                    # 学習結果
│   └── {model_path}/
│       ├── sft_model/          # SFTアダプター
│       └── merged_model/       # マージ済みモデル
└── llama.cpp/                  # llama.cppサブモジュール（GGUF変換用）
```

## 🎓 学習について

### 学習データの準備

1. [つくよみちゃん会話AI育成計画](https://tyc.rei-yumesaki.net/material/kaiwa-ai/)からデータをダウンロード
2. `download_and_convert.py` を実行して `tsukuyomi.jsonl` 形式に変換
3. データは `./data/tsukuyomi.jsonl` に配置されます

### 学習技術

- **QLoRA**: 4bit量子化によるメモリ効率的な学習
- **LoRA**: パラメータ効率的ファインチューニング
- **SFT**: Supervised Fine-Tuning
- **Gradient Checkpointing**: メモリ使用量の削減

### 学習パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `--model_path` | `sbintuitions/sarashina2.2-3b-instruct-v0.1` | ベースモデルのパス |
| `--data_path` | `./data/tsukuyomi.jsonl` | 学習データのパス |
| `--num_epochs` | 3 | エポック数 |
| `--batch_size` | 2 | バッチサイズ |
| `--learning_rate` | 2e-4 | 学習率 |
| `--max_length` | 4096 | 最大シーケンス長 |
| `--gradient_accumulation_steps` | 4 | 勾配累積ステップ数 |
| `--lora_r` | 64 | LoRAのランク |
| `--lora_alpha` | 128 | LoRAのアルファ値 |
| `--lora_dropout` | 0.1 | LoRAのドロップアウト率 |

### 出力パス

学習結果は以下のパスに自動的に保存されます：

- **SFTアダプター**: `./results/{model_path}/sft_model/`
- **マージ済みモデル**: `./results/{base_model}/merged_model/`

`--base_model` を指定すると、`adapter_path` と `output_path` は自動的に生成されます。

## 🎯 使用方法

### データのダウンロードと変換

```bash
uv run download_and_convert.py
```

このスクリプトは：
- 公式配布ページからデータをダウンロード
- JSONL形式に変換
- `./data/tsukuyomi.jsonl` に保存

### SFT学習

```bash
uv run sft_train.py [オプション]
```

**主なオプション:**
- `--model_path`: ベースモデルのパス（デフォルト: `sbintuitions/sarashina2.2-3b-instruct-v0.1`）
- `--data_path`: 学習データのパス（デフォルト: `./data/tsukuyomi.jsonl`）
- `--num_epochs`: エポック数（デフォルト: 3）
- `--batch_size`: バッチサイズ（デフォルト: 2）
- `--learning_rate`: 学習率（デフォルト: 2e-4）
- `--lora_r`: LoRAのランク（デフォルト: 64）
- `--lora_alpha`: LoRAのアルファ値（デフォルト: 128）

### モデルマージ

```bash
uv run merge_model.py [オプション]
```

**主なオプション:**
- `--base_model`: ベースモデルのパス（デフォルト: `sbintuitions/sarashina2.2-3b-instruct-v0.1`）
- `--device`: 使用するデバイス（デフォルト: `auto`）

`adapter_path` と `output_path` は `base_model` から自動的に生成されます。

## 技術仕様

* **ベースモデル**: Sarashina2.2 3B Instruct v0.1
* **学習データ**: つくよみちゃん会話AI育成計画
* **ファインチューニング**: PEFT (LoRA + QLoRA)
* **量子化**: QLoRA 4bit (NF4)
* **最大コンテキスト**: 4,096トークン
* **学習フレームワーク**: TRL (SFTTrainer)
* **依存関係管理**: uv

---

## 🙏 クレジット

### 🔹 ベースモデル（構造・学習済み重み・推論コード）

* [Sarashina2.2 3B Instruct v0.1](https://huggingface.co/sbintuitions/sarashina2.2-3b-instruct-v0.1) © SB Intuitions
  * ライセンス: [MIT License](https://opensource.org/licenses/MIT)

### 🔸 学習データ（追加学習した会話テキストデータ）

* [つくよみちゃん会話AI育成計画](https://tyc.rei-yumesaki.net/material/kaiwa-ai/)
* ライセンス: [つくよみちゃん会話AI育成計画の利用規約](https://tyc.rei-yumesaki.net/material/kaiwa-ai/)

### キャラクター

* [フリー素材キャラクター「つくよみちゃん」](https://tyc.rei-yumesaki.net) © Rei Yumesaki
* ライセンス: [つくよみちゃんキャラクターライセンス](https://tyc.rei-yumesaki.net/material/kaiwa-ai/)

---

## 📄 本モデルの利用規約

### ⚠️ 禁止事項

以下の目的での利用を禁止します（定義は [つくよみちゃんキャラクターライセンス](https://tyc.rei-yumesaki.net/about/terms/) に準拠）：

1. 特定の人物や団体を批判・攻撃すること
2. 政治的立場・宗教・思想への賛同または反対を呼びかけること
3. 刺激の強い表現をゾーニングなしで公開すること

### スクリーンショット・動画投稿等について

本モデルの動作画面のスクリーンショットやキャプチャ動画を投稿する場合、または生成された会話を元ネタとする作品を公開する場合は、以下のクレジットを記載してください：

- **モデル名**: 使用したファインチューニング済みモデルの名前（例: `yukiharada1228/tsukuyomichan-sarashina2.2-3b-instruct-v0.1-GGUF`）
- **キャラクター**: つくよみちゃん

### 📌 改変・再配布等について

以下の場合、つくよみちゃん会話AI育成計画に由来する部分については、[つくよみちゃん会話AI育成計画の利用規約](https://tyc.rei-yumesaki.net/material/kaiwa-ai/) に従ってください：

* 本モデルを**再配布**する場合（**改変の有無を問いません**）
* 本モデルから生成された会話を素材として配布する場合
* 本モデルから生成された会話を利用して新たな会話AIを作成する場合

---

## 🛡 免責事項

* このモデルは「現状有姿 (AS IS)」で提供されます。生成される内容の正確性・有用性は保証されません。
* 本モデルの利用により生じたいかなる損害についても、開発者は責任を負いません。
* つくよみちゃん会話AI育成計画に関する免責事項は [つくよみちゃんキャラクターライセンスの免責事項](https://tyc.rei-yumesaki.net/about/terms/#disclaimer) をご確認ください。

---

## 関連リンク

**ベースモデル:**

* [Sarashina2.2 3B Instruct v0.1](https://huggingface.co/sbintuitions/sarashina2.2-3b-instruct-v0.1)

**学習データ:**

* [つくよみちゃん会話AI育成計画](https://tyc.rei-yumesaki.net/material/kaiwa-ai/)
* [つくよみちゃん公式サイト](https://tyc.rei-yumesaki.net)

**このプロジェクト:**

* [プロジェクトリポジトリ](https://github.com/yukiharada1228/tsukuyomichan-llm-ft)
* [問題・質問 (GitHub Issues)](https://github.com/yukiharada1228/tsukuyomichan-llm-ft/issues)
