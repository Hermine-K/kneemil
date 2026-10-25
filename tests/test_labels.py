import numpy as np
import pytest

from kneemil.constants import TARGETS
from kneemil.labels import (
    KEYS,
    NOT_MENTIONED,
    SOFT,
    SYSTEM_PROMPT,
    generate_safe,
    make_batches,
    normalize_category,
    parse_answer,
    to_soft,
)

ANSWER = ('```json\n{"ACL":"definite","MCL":"negative","Medial Meniscus":"tear","Lateral Meniscus":"not mentioned",'
          '"Medial OA":"possible","Lateral OA":"no evidence","PF OA":"negative","Effusion":"probable",'
          '"Synovitis":"not_mentioned","Bakers":"present","Contusion":"negative","Fracture":"negative"}\n```')


def test_prompt_lists_every_finding():
    for k in KEYS:
        assert f'"{k}"' in SYSTEM_PROMPT
    assert "LCA" in SYSTEM_PROMPT  # multilingual glossary


@pytest.mark.parametrize("value,expected", [
    ("definite", "definite"), ("Present", "definite"), ("tear", "definite"), ("osteoarthritis", "definite"),
    ("suspected", "probable"), ("questionable", "possible"),
    ("not mentioned", "not_mentioned"), ("absent", "negative"), ("no tear", "negative"),
    ("intact", "negative"), ("", None),
])
def test_normalize_category(value, expected):
    assert normalize_category(value) == expected


def test_parse_answer_handles_fences_and_free_words():
    cats = parse_answer(ANSWER)
    assert cats[:5] == ["definite", "negative", "definite", "not_mentioned", "possible"]
    assert cats[5] == "negative" and cats[9] == "definite"


def test_parse_answer_truncated_json():
    truncated = ANSWER.split('"Contusion"')[0]
    cats = parse_answer(truncated)
    assert cats is not None and cats[-1] == "not_mentioned"


def test_parse_answer_rejects_garbage():
    assert parse_answer("I cannot answer.") is None
    assert parse_answer('{"ACL": "definite"}') is None  # too few keys


def test_to_soft():
    soft = to_soft(["definite", "probable", "possible", "negative", "not_mentioned", None] + ["negative"] * 6)
    assert soft[:4].tolist() == [SOFT["definite"], SOFT["probable"], SOFT["possible"], SOFT["negative"]]
    assert soft[4] == soft[5] == NOT_MENTIONED
    assert len(soft) == len(TARGETS)


def test_make_batches_respect_token_budget():
    rng = np.random.RandomState(0)
    lengths = [(i, int(L)) for i, L in enumerate(rng.randint(300, 3300, 200))]
    batches = make_batches(lengths, token_budget=14000, max_batch=16)
    L = dict(lengths)
    assert sorted(i for b in batches for i in b) == list(range(200))
    for b in batches:
        assert len(b) <= 16
        assert len(b) == 1 or len(b) * max(L[i] for i in b) <= 14000


def test_generate_safe_splits_on_out_of_memory():
    calls = []

    def fake_generate(prompts):
        calls.append(len(prompts))
        if len(prompts) > 2:
            raise RuntimeError("CUDA out of memory")
        return [p.upper() for p in prompts]

    out = generate_safe(fake_generate, list("abcdefg"))
    assert out == list("ABCDEFG")
    assert calls[0] == 7 and max(calls[1:]) <= 4


def test_generate_safe_reraises_other_errors():
    def broken(prompts):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        generate_safe(broken, ["x"])
