import argparse
import json
import logging
import os
import sys

import torch
from datasets import Dataset
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


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


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # ログ設定の初期化
    logger = setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="sbintuitions/sarashina2.2-3b-instruct-v0.1"
    )
    parser.add_argument("--data_path", type=str, default="./data/tsukuyomi.jsonl")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = os.path.join("./results", args.model_path, "sft_model")

    # 学習設定の表示
    logger.info("=" * 60)
    logger.info("SFTトレーニング開始（QLoRA使用）")
    logger.info("=" * 60)
    logger.info(f"モデルパス: {args.model_path}")
    logger.info(f"データパス: {args.data_path}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    logger.info(f"エポック数: {args.num_epochs}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"学習率: {args.learning_rate}")
    logger.info(
        f"QLoRA設定: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
    )
    logger.info("=" * 60)

    # データファイルの存在確認
    if not os.path.exists(args.data_path):
        logger.error(f"データファイルが見つかりません: {args.data_path}")
        sys.exit(1)

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"出力ディレクトリを作成しました: {output_dir}")

    # トークナイザー
    logger.info("トークナイザーを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("トークナイザーの読み込み完了")

    # QLoRA用量子化
    logger.info("モデルを読み込み中（QLoRA用量子化）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    logger.info("モデルの読み込み完了")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA設定を適用完了")

    # データセット
    logger.info("データセットを読み込み中...")
    data = load_jsonl(args.data_path)
    dataset = Dataset.from_list(data)
    logger.info(f"データセット読み込み完了（サンプル数: {len(dataset)}）")

    # SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=True,
        report_to=None,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        max_length=args.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    logger.info("トレーニングを開始します...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info("=" * 60)
    logger.info("SFTトレーニング完了")
    logger.info("=" * 60)
    logger.info(f"トレーニング済みモデル: {output_dir}")


if __name__ == "__main__":
    main()
