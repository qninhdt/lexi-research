import pytest

from lexi_research.format import BandConfig, default_config_path
from lexi_research.teacher.schemas import SenseRef
from serve.service import GradeUnavailable, grade


class Backend:
    def __init__(self, replies):
        self.replies = iter(replies)

    async def grade(self, target, sense, text):
        return next(self.replies)


async def test_grade_derives_all_five_fields() -> None:
    result = await grade(
        Backend(
            [
                {
                    "correction": "The room [have>has:agr] light.",
                    "meaning": 4,
                    "feedback": "Good use.",
                }
            ]
        ),
        "light",
        SenseRef(definition="brightness", pos="noun"),
        "The room have light.",
        BandConfig.from_json(default_config_path()),
    )
    assert (result.meaning, result.grammar, result.naturalness, result.retries) == (4, 1, 4, 0)


async def test_invalid_output_retries_then_fails() -> None:
    with pytest.raises(GradeUnavailable):
        await grade(
            Backend([{"correction": "changed", "meaning": 4, "feedback": "Good."}] * 2),
            "light",
            SenseRef(definition="brightness", pos="noun"),
            "The room has light.",
            BandConfig.from_json(default_config_path()),
            retries=1,
        )
