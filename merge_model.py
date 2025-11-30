import argparse
import json
import logging
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_logging():
    """
    ログ設定を初期化
    """
    # ログフォーマットの設定
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # ログレベルの設定（標準出力のみ）
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


def load_adapter_config(adapter_path):
    """
    アダプター設定を読み込み
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"アダプター設定ファイルが見つかりません: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_model_and_adapter(base_model_path, adapter_path, output_path, device="auto"):
    """
    ベースモデルとLoRAアダプターをマージ
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ベースモデルとLoRAアダプターのマージ開始")
    logger.info("=" * 60)
    logger.info(f"ベースモデル: {base_model_path}")
    logger.info(f"アダプター: {adapter_path}")
    logger.info(f"出力先: {output_path}")
    logger.info(f"デバイス: {device}")
    logger.info("=" * 60)

    # アダプターパスの存在確認
    if not os.path.exists(adapter_path):
        logger.error(f"アダプターパスが見つかりません: {adapter_path}")
        raise FileNotFoundError(f"アダプターパスが見つかりません: {adapter_path}")

    # アダプター設定を読み込み
    adapter_config = load_adapter_config(adapter_path)
    logger.info(
        f"アダプター設定: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}"
    )

    # トークナイザー
    logger.info("トークナイザーを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("トークナイザーの読み込み完了")

    # ベースモデルを読み込み
    logger.info("ベースモデルを読み込み中...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    logger.info("ベースモデルの読み込み完了")

    # LoRAアダプターを読み込み
    logger.info("LoRAアダプターを読み込み中...")
    model = PeftModel.from_pretrained(model, adapter_path)
    logger.info("LoRAアダプターの読み込み完了")

    # アダプターをマージ
    logger.info("アダプターをベースモデルにマージ中...")
    model = model.merge_and_unload()
    logger.info("マージ完了")

    # 出力ディレクトリの作成
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"出力ディレクトリを作成しました: {output_path}")

    # マージされたモデルを保存
    logger.info("マージされたモデルを保存中...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    logger.info(f"モデルを保存しました: {output_path}")

    # モデル情報を保存
    model_info = {
        "base_model": base_model_path,
        "adapter_path": adapter_path,
        "merged_model": output_path,
        "adapter_config": adapter_config,
        "model_type": "merged_lora",
        "description": "ベースモデルとLoRAアダプターをマージしたモデル",
    }

    info_path = os.path.join(output_path, "model_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("マージ完了")
    logger.info("=" * 60)
    logger.info(f"マージされたモデル: {output_path}")


def main():
    # ログ設定の初期化
    logger = setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="sbintuitions/sarashina2.2-3b",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="使用するデバイス (auto, cpu, cuda:0, etc.)",
    )

    args = parser.parse_args()

    # アダプターパスの自動生成
    adapter_path = os.path.join("./results", args.base_model, "sft_model")

    # 出力パスの自動生成
    output_path = os.path.join("./results", args.base_model, "merged_model")

    try:
        # マージ実行
        merged_path = merge_model_and_adapter(
            base_model_path=args.base_model,
            adapter_path=adapter_path,
            output_path=output_path,
            device=args.device,
        )

    except Exception as e:
        logger.error(f"マージに失敗しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
