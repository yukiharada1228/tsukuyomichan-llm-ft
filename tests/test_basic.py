"""
基本的な動作テスト
"""

import json
from unittest.mock import mock_open


def test_main_import():
    """main.pyがインポートできること"""
    import main

    assert hasattr(main, "main")


def test_download_and_convert_import():
    """download_and_convert.pyがインポートできること"""
    import download_and_convert

    assert hasattr(download_and_convert, "main")
    assert hasattr(download_and_convert, "setup_logging")
    assert hasattr(download_and_convert, "download_tsukuyomi_data")
    assert hasattr(download_and_convert, "convert_to_jsonl")


def test_sft_train_import():
    """sft_train.pyがインポートできること"""
    import sft_train

    assert hasattr(sft_train, "main")
    assert hasattr(sft_train, "setup_logging")
    assert hasattr(sft_train, "load_jsonl")


def test_merge_model_import():
    """merge_model.pyがインポートできること"""
    import merge_model

    assert hasattr(merge_model, "main")
    assert hasattr(merge_model, "setup_logging")
    assert hasattr(merge_model, "merge_model_and_adapter")


class TestDownloadAndConvert:
    """download_and_convert.pyの基本テスト"""

    def test_setup_logging(self):
        """ロガーが正しく設定されること"""
        from download_and_convert import setup_logging

        logger = setup_logging()
        assert logger is not None

    def test_cleanup_temp_files(self, mocker):
        """一時ファイルのクリーンアップ処理が動作すること"""
        from download_and_convert import cleanup_temp_files, setup_logging

        mocker.patch("os.path.exists", return_value=False)
        logger = setup_logging()

        cleanup_temp_files(logger)


class TestSftTrain:
    """sft_train.pyの基本テスト"""

    def test_load_jsonl(self, mocker):
        """JSONLファイルの読み込みが動作すること"""
        from sft_train import load_jsonl

        test_data = [{"messages": [{"role": "user", "content": "test"}]}]
        jsonl_content = json.dumps(test_data[0])

        m = mock_open(read_data=jsonl_content)
        mocker.patch("builtins.open", m)

        result = load_jsonl("test.jsonl")
        assert len(result) == 1


class TestMergeModel:
    """merge_model.pyの基本テスト"""

    def test_load_adapter_config(self, mocker, tmp_path):
        """アダプター設定の読み込みが動作すること"""
        from merge_model import load_adapter_config

        config_data = {"r": 64, "lora_alpha": 128}
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config_file = adapter_dir / "adapter_config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_adapter_config(str(adapter_dir))
        assert result["r"] == 64
