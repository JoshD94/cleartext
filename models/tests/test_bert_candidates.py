import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.bert_candidates import paired_input, proposals, masked_predictions
from cleartext.features import nlp


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)


def test_pair_retains_original_and_masks_exact_repeated_target(tokenizer):
    text = "The bank is beside the river bank."
    start = text.rindex("bank")
    ids, types, position = paired_input(tokenizer, text, start, start + 4)
    assert ids[position] == tokenizer.mask_token_id and types[position] == 1
    first = [piece for piece, segment in zip(ids, types) if segment == 0]
    assert first == tokenizer(text).input_ids
    second = tokenizer.decode(
        [piece for piece, segment in zip(ids, types) if segment == 1]
    )
    assert second.count("bank") == 1 and second.count("[MASK]") == 1
    long = "record " * 400 + "unaffordable" + " record" * 400
    begin = long.index("unaffordable")
    ids, _, position = paired_input(tokenizer, long, begin, begin + 12)
    assert len(ids) <= 512 and ids[position] == tokenizer.mask_token_id


def test_inflection_deduplication_and_no_invented_sense_for_antonym():
    token = next(
        token
        for token in nlp()("The medication alleviates pain.")
        if token.text == "alleviates"
    )
    raw = [
        {"token": word, "rank": rank + 1, "probability": 0.1}
        for rank, word in enumerate(
            ["relieve", "relieves", "worsen", "##ing", ".", "alleviate"]
        )
    ]
    _, candidates = proposals(token, raw)
    by_word = {candidate["word"]: candidate for candidate in candidates}
    assert list(by_word).count("relieves") == 1
    assert by_word["relieves"]["senses"]
    assert by_word["worsens"]["senses"] == []
    assert by_word["worsens"]["source"] == "bert"
    assert "alleviates" not in by_word
    assert all(candidate["word"].isalpha() for candidate in candidates)


def test_empty_batch_needs_no_encoder():
    assert masked_predictions([]) == []
    with pytest.raises(ValueError):
        masked_predictions([], top_k=0)
