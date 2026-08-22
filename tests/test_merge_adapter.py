"""Tests for PEFT adapter merging utilities."""

from unittest.mock import MagicMock

from tau_research.training.merge_adapter import merge_lora_adapter


def test_merge_lora_adapter_calls() -> None:
    mock_peft_model = MagicMock()
    mock_merged_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_merged_model

    mock_tokenizer = MagicMock()

    merge_lora_adapter(
        peft_model=mock_peft_model,
        tokenizer=mock_tokenizer,
        output_dir="artifacts/models/test_merged",
    )

    mock_peft_model.merge_and_unload.assert_called_once()
    mock_merged_model.save_pretrained.assert_called_once_with("artifacts/models/test_merged")
    mock_tokenizer.save_pretrained.assert_called_once_with("artifacts/models/test_merged")
