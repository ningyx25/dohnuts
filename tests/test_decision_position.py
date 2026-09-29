"""The decision readout sits at the stem end and never touches a candidate."""

import pytest
import torch

from dohnuts.predictor import (
    _placeholder_shift,
    _stem_end_token,
    locate_decision_positions,
    render,
    render_question,
)

STATE = {"screen": "settings page"}


class CharTokenizer:
    """One character per token, plus the reserved special tokens."""

    padding_side = "right"
    specials = {
        "<|fim_suffix|>": 100,
        "<|vision_start|>": 101,
        "<|image_pad|>": 102,
        "<|vision_end|>": 103,
    }

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        index = 0
        while index < len(text):
            for token, token_id in self.specials.items():
                if text.startswith(token, index):
                    ids.append(token_id)
                    # Special tokens carry no character span, as in the real tokenizer.
                    offsets.append((0, 0))
                    index += len(token)
                    break
            else:
                ids.append(ord(text[index]) + 1000)
                offsets.append((index, index + 1))
                index += 1
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class StubProcessor:
    def __init__(self, expansion=3):
        self.tokenizer = CharTokenizer()
        self.image_token_id = 102
        self.expansion = expansion


def question(options):
    return {"type": "choice", "instructions": "pick one", "criteria": options}


def processed_row(processor, text):
    """Reproduce the processor expanding one image placeholder into many tokens."""
    ids = processor.tokenizer(text)["input_ids"]
    expanded = []
    for token in ids:
        expanded.extend(
            [token] * processor.expansion if token == processor.image_token_id else [token]
        )
    return torch.tensor(expanded)


def package(text, processor):
    encoded = processor.tokenizer(text, return_offsets_mapping=True)
    return encoded, processed_row(processor, text)


def pad_rows(rows):
    """Right padding, exactly as the collator's processor call produces it."""
    width = max(len(row) for row in rows)
    return torch.stack(
        [torch.cat([row, torch.zeros(width - len(row), dtype=torch.long)]) for row in rows]
    )


def render_case(options, has_image=False, state=None):
    return render_question(render(state or STATE), question(options), has_image=has_image)


def test_decision_token_is_the_stem_end_character():
    processor = StubProcessor()
    content, labels, stem_end = render_case(["alpha", "beta"])
    assert labels == ["alpha", "beta"]
    encoded, row = package(content, processor)
    position = locate_decision_positions(
        processor, [content], [stem_end], input_ids=row.unsqueeze(0)
    )[0]
    raw_index = _stem_end_token(encoded["offset_mapping"], stem_end)
    start, end = encoded["offset_mapping"][raw_index]
    assert end == stem_end
    assert content[start:end] == "\n"
    assert int(row[position]) == encoded["input_ids"][raw_index]


def test_decision_position_ignores_candidate_descriptions():
    processor = StubProcessor()
    short = render_case(["a", "b"])
    long = render_case(["a very long candidate description", "another one"])
    assert short[2] == long[2]
    row = pad_rows([processed_row(processor, short[0]), processed_row(processor, long[0])])
    positions = locate_decision_positions(
        processor, [short[0], long[0]], [short[2], long[2]], input_ids=row
    )
    assert positions[0] == positions[1]
    assert int(row[0, positions[0]]) == int(row[1, positions[1]])


def test_image_prefix_shifts_the_decision_index_back():
    processor = StubProcessor(expansion=5)
    content, _, stem_end = render_case(["alpha", "beta"], has_image=True)
    encoded, row = package(content, processor)
    raw_index = _stem_end_token(encoded["offset_mapping"], stem_end)
    shift = _placeholder_shift(encoded["input_ids"], row, processor.image_token_id)
    assert shift == processor.expansion - 1
    position = locate_decision_positions(
        processor, [content], [stem_end], input_ids=row.unsqueeze(0)
    )[0]
    assert int(position) == raw_index + shift
    assert int(row[position]) == encoded["input_ids"][raw_index]
    assert encoded["offset_mapping"][raw_index][1] == stem_end


def test_right_padding_keeps_row_indices_inside_a_batch():
    processor = StubProcessor()
    cases = [render_case(["alpha", "beta"]), render_case(["gamma", "delta", "epsilon"])]
    rows = [processed_row(processor, content) for content, _, _ in cases]
    padded = pad_rows(rows)
    texts = [content for content, _, _ in cases]
    ends = [stem_end for _, _, stem_end in cases]
    positions = locate_decision_positions(processor, texts, ends, input_ids=padded)
    for index, row in enumerate(rows):
        single = locate_decision_positions(
            processor, texts[index : index + 1], ends[index : index + 1], input_ids=row.unsqueeze(0)
        )
        assert positions[index] == single[0]
        assert int(padded[index, positions[index]]) == int(row[single[0]])


def test_mixed_image_and_text_rows_keep_their_own_shift():
    processor = StubProcessor(expansion=2)
    with_image = render_case(["alpha", "beta"], has_image=True)
    text_only = render_case(["gamma", "delta"])
    padded = pad_rows(
        [processed_row(processor, with_image[0]), processed_row(processor, text_only[0])]
    )
    positions = locate_decision_positions(
        processor,
        [with_image[0], text_only[0]],
        [with_image[2], text_only[2]],
        input_ids=padded,
    )
    assert int(padded[0, positions[0]]) == processor.tokenizer("\n")["input_ids"][0]
    assert int(padded[1, positions[1]]) == processor.tokenizer("\n")["input_ids"][0]


def test_left_padded_tokenizer_is_rejected():
    processor = StubProcessor()
    processor.tokenizer.padding_side = "left"
    content, _, stem_end = render_case(["alpha", "beta"])
    row = processed_row(processor, content).unsqueeze(0)
    with pytest.raises(ValueError, match="right-padded"):
        locate_decision_positions(processor, [content], [stem_end], input_ids=row)


def test_drifted_prompt_is_rejected():
    processor = StubProcessor()
    content, _, stem_end = render_case(["alpha", "beta"])
    # A longer state shifts every character, so the recorded index no longer
    # holds the stem token the offsets described.
    other, _, _ = render_case(["gamma", "delta"], state={"screen": "x" * 50})
    row = processed_row(processor, other).unsqueeze(0)
    with pytest.raises(ValueError, match="drifted"):
        locate_decision_positions(processor, [content], [stem_end], input_ids=row)


def test_a_stem_without_a_usable_token_is_rejected():
    with pytest.raises(ValueError, match="no decision token"):
        _stem_end_token([(0, 0), (0, 0)], 10)
