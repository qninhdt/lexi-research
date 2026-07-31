from lexi_research.data.process import process_rows
from lexi_research.format import BandConfig, default_config_path


def test_process_preserves_invalid_rows_and_derives_fields() -> None:
    rows = [
        {"req_uid": "a", "target_norm": "bright", "text": "Bright room.", "correction": "Bright room.", "meaning": 4, "feedback": "Good sentence."},
        {"req_uid": "b", "target_norm": "dark", "text": "Dark room.", "correction": "changed text.", "meaning": 4, "feedback": "Good sentence."},
    ]
    clean, rejects, report = process_rows(rows, BandConfig.from_json(default_config_path()), seed=1, version="v1", max_stratum_share=1)
    assert len(clean) == 1 and clean[0]["grammar"] == 4
    assert rejects[0]["reject_reason"] == "text_altered"
    assert report["rejected_rows"] == 1
