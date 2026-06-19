from pathlib import Path


def test_answer_file() -> None:
    assert Path('/app/answer.txt').read_text().strip() == 'ok'
