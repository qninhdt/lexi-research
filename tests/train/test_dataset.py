from lexi_research.teacher import render_grader_prompt
from lexi_research.teacher.schemas import SenseRef
from lexi_research.train.dataset import training_example


def test_training_prompt_uses_the_serving_renderer() -> None:
    row = {
        "target": "bright",
        "definition": "full of light",
        "pos": "adjective",
        "text": "The room is bright.",
        "correction": "The room is bright.",
        "meaning": 4,
        "feedback": "Good sentence.",
    }
    example = training_example(row, nonce="fixed")
    messages = render_grader_prompt(
        "bright",
        SenseRef(definition="full of light", pos="adjective"),
        "The room is bright.",
        nonce="fixed",
    )
    assert (example["system"], example["user"]) == (messages[0]["content"], messages[1]["content"])
    assert "grammar" not in example["completion"]
