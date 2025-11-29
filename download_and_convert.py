"""
つくよみちゃん会話AI育成計画データ変換スクリプト

このスクリプトは公式配布ページからデータをダウンロードし、
学習用のJSONL形式に変換します。

利用規約に従い、元データの再配布は行わず、
ユーザーが各自で公式データをダウンロードして使用できる仕組みを提供します。

元データ: https://tyc.rei-yumesaki.net/material/kaiwa-ai/
"""

import json
import logging
import os
import sys

import pandas as pd
import requests


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
        handlers=[logging.StreamHandler(sys.stdout)],  # コンソール出力のみ
    )

    return logging.getLogger(__name__)


def download_tsukuyomi_data(logger):
    """
    公式配布ページからつくよみちゃんデータをダウンロード
    """
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS9Jxjsp_H1v-N37xHu6LOZ2vOpedtgSe6VGht_-kL_-PWBEJ3c36ofAm8PMT0KOA/pub?output=xlsx"
    output_file = "./data/tsukuyomi.jsonl"

    # JSONLファイルが既に存在するかチェック
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        logger.info(
            f"既存のJSONLファイルが見つかりました: {output_file} (ファイルサイズ: {file_size:,} bytes)"
        )
        logger.info("データ変換が既に完了しているため、ダウンロードをスキップします")
        return True

    logger.info("つくよみちゃん会話AI育成計画の公式データをダウンロード開始")
    logger.debug(f"ダウンロードURL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        # 一時的なExcelファイルとして保存
        temp_file = "tsukuyomi_temp.xlsx"
        with open(temp_file, "wb") as f:
            f.write(response.content)

        file_size = len(response.content)
        logger.info(
            f"データのダウンロードが完了しました (ファイルサイズ: {file_size:,} bytes)"
        )
        logger.debug(f"一時ファイル保存先: {temp_file}")
        return True

    except requests.RequestException as e:
        logger.error(f"データのダウンロードに失敗しました: {e}")
        return False


def convert_to_jsonl(logger):
    """
    ExcelデータをJSONL形式に変換
    """
    logger.info("データをJSONL形式に変換開始")

    try:
        # Excelファイルを読み込み
        temp_file = "tsukuyomi_temp.xlsx"
        logger.debug(f"Excelファイル読み込み: {temp_file}")

        # ヘッダー行をスキップして読み込み
        df = pd.read_excel(temp_file, header=None)
        logger.info(f"Excelファイル読み込み完了 (行数: {len(df)})")

        # 出力用のJSONLデータ
        jsonl_data = []
        processed_count = 0
        skipped_count = 0

        # データ構造を分析
        logger.info(f"Excelファイルの列数: {len(df.columns)}")
        logger.info(f"Excelファイルの行数: {len(df)}")

        # 最初の数行を確認
        logger.info("最初の5行の内容:")
        for i in range(min(5, len(df))):
            row_data = []
            for j in range(len(df.columns)):
                cell_value = df.iloc[i, j]
                if pd.notna(cell_value):
                    row_data.append(f"列{j}: '{str(cell_value)[:50]}'")
                else:
                    row_data.append(f"列{j}: 空")
            logger.info(f"行{i}: {', '.join(row_data)}")

        # 各行を処理（最初の2行はスキップ）
        for index, row in df.iterrows():
            # 最初の2行（説明文とヘッダー行）をスキップ
            if index <= 1:
                if index == 0:
                    logger.info(f"行{index+1}: 説明文をスキップ")
                else:
                    logger.info(f"行{index+1}: ヘッダー行をスキップ")
                skipped_count += 1
                continue
            # 質問と回答のペアを探す
            # 1往復の会話: B列（質問）とC列（回答）のペア
            if pd.notna(row.iloc[1]) and pd.notna(row.iloc[2]) and pd.isna(row.iloc[3]):
                conversation = {
                    "messages": [
                        {"role": "system", "content": "あなたは、つくよみちゃんです。"},
                        {"role": "user", "content": str(row.iloc[1]).strip()},
                        {"role": "assistant", "content": str(row.iloc[2]).strip()},
                    ]
                }
                jsonl_data.append(conversation)
                processed_count += 1
                logger.debug(f"行 {index+1}: 1往復会話（B列→C列）を処理")

            # 2往復の会話: A列→B列→C列→D列の4つのメッセージ
            elif (
                pd.notna(row.iloc[1])
                and pd.notna(row.iloc[2])
                and pd.notna(row.iloc[3])
                and pd.notna(row.iloc[4])
            ):
                conversation = {
                    "messages": [
                        {"role": "system", "content": "あなたは、つくよみちゃんです。"},
                        {"role": "user", "content": str(row.iloc[1]).strip()},
                        {"role": "assistant", "content": str(row.iloc[2]).strip()},
                        {"role": "user", "content": str(row.iloc[3]).strip()},
                        {"role": "assistant", "content": str(row.iloc[4]).strip()},
                    ]
                }
                jsonl_data.append(conversation)
                processed_count += 1
                logger.debug(f"行 {index+1}: 2往復会話（A列→B列→C列→D列）を処理")

            else:
                skipped_count += 1
                logger.debug(f"行 {index+1}: スキップ（有効な質問・回答ペアなし）")

        # JSONLファイルに保存
        output_dir = "./data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "tsukuyomi.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in jsonl_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            f"変換完了: {processed_count}件の会話データをJSONL形式で保存しました"
        )
        logger.info(f"スキップされた行数: {skipped_count}件")
        logger.debug(f"出力ファイル: {output_file}")
        return True, output_file

    except Exception as e:
        logger.error(f"データの変換に失敗しました: {e}")
        logger.exception("詳細なエラー情報:")
        return False, None


def cleanup_temp_files(logger):
    """
    一時ファイルを削除
    """
    temp_files = ["tsukuyomi_temp.xlsx"]

    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"一時ファイル {file} を削除しました")
            except OSError as e:
                logger.warning(f"一時ファイル {file} の削除に失敗しました: {e}")
        else:
            logger.debug(f"一時ファイル {file} は存在しません")


def main():
    """
    メイン処理
    """
    # ログ設定の初期化
    logger = setup_logging()

    # データダウンロード
    if not download_tsukuyomi_data(logger):
        logger.error("データダウンロードに失敗したため、処理を終了します")
        sys.exit(1)

    # JSONLファイルが既に存在するかチェック
    output_file = "./data/tsukuyomi.jsonl"
    if os.path.exists(output_file):
        logger.info("既存のJSONLファイルが存在するため、変換処理をスキップします")
        logger.info("処理を終了します")
        sys.exit(0)

    # JSONL形式に変換
    success, output_file = convert_to_jsonl(logger)
    if not success:
        logger.error("データ変換に失敗したため、一時ファイルを削除して終了します")
        cleanup_temp_files(logger)
        sys.exit(1)

    # 一時ファイルの削除
    cleanup_temp_files(logger)


if __name__ == "__main__":
    main()
