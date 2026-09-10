from __future__ import annotations

from types import SimpleNamespace

import torch

from chcons.agent import ReActAgent
from chcons.uld import ULDLogitsProcessor, combine_uld_logits


class _Batch(dict):
    def to(self, device: torch.device) -> _Batch:
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    eos_token_id = 0

    def __call__(self, _text: str, **kwargs) -> _Batch:
        assert kwargs == {
            "return_tensors": "pt",
            "add_special_tokens": False,
            "return_token_type_ids": False,
        }
        return _Batch({"input_ids": torch.tensor([[4, 5]])})

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return ",".join(str(token_id) for token_id in token_ids.tolist())


class _SeededGenerateModel:
    device = torch.device("cpu")

    def generate(self, **kwargs) -> torch.Tensor:
        input_ids = kwargs.pop("input_ids")
        assert kwargs == {
            "max_new_tokens": 3,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": 0,
            "logits_processor": None,
        }
        sampled = torch.randint(10, 50, (1, 3))
        return torch.cat((input_ids, sampled), dim=1)


class _KnownLogitsModel:
    device = torch.device("cpu")

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.seen_input_ids: list[torch.Tensor] = []

    def __call__(self, **kwargs) -> SimpleNamespace:
        input_ids = kwargs.pop("input_ids")
        self.seen_input_ids.append(input_ids.clone())
        attention_mask = kwargs.pop("attention_mask")
        assert attention_mask.shape[1] >= input_ids.shape[1]
        past_key_values = kwargs.pop("past_key_values")
        if len(self.seen_input_ids) == 1:
            assert past_key_values is None
        else:
            assert past_key_values == ("stub-cache",)
        assert kwargs == {
            "use_cache": True,
            "return_dict": True,
        }
        return SimpleNamespace(logits=self.logits, past_key_values=("stub-cache",))


def test_no_assistant_generation_matches_pre_change_decode_with_fixed_seed() -> None:
    """The shared no-assistant path still executes the pre-change generate call exactly."""
    tokenizer = _Tokenizer()
    model = _SeededGenerateModel()
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        retriever=SimpleNamespace(),
        max_new_tokens=3,
    )
    assert agent.logits_processors == []

    torch.manual_seed(9182)
    legacy_inputs = tokenizer(
        "prompt",
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to(model.device)
    with torch.no_grad():
        legacy_output_ids = model.generate(
            **legacy_inputs,
            max_new_tokens=3,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=None,
        )
    expected_ids = legacy_output_ids[0, legacy_inputs["input_ids"].shape[1] :]

    torch.manual_seed(9182)
    actual_text = agent._generate_block("prompt")

    assert actual_text == tokenizer.decode(expected_ids, skip_special_tokens=True)
    assert actual_text == "12,32,16"


def test_uld_combines_known_logits_and_changes_argmax() -> None:
    base_next_logits = torch.tensor([[3.0, 2.0, 1.0]])
    assistant_next_logits = torch.tensor([[5.0, 0.0, 0.0]])
    base = _KnownLogitsModel(
        torch.stack((torch.zeros_like(base_next_logits), base_next_logits), dim=1)
    )
    assistant = _KnownLogitsModel(
        torch.stack((torch.zeros_like(assistant_next_logits), assistant_next_logits), dim=1)
    )
    processor = ULDLogitsProcessor(assistant_model=assistant, weight=-1.0)
    input_ids = torch.tensor([[7, 8]])
    base_logits = base(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ).logits[:, -1, :]

    combined = processor(input_ids, base_logits)

    expected = torch.tensor([[-2.0, 2.0, 1.0]])
    assert torch.equal(combined, expected)
    assert torch.equal(combined, combine_uld_logits(base_logits, assistant_next_logits, -1.0))
    assert base_logits.argmax(dim=-1).item() == 0
    assert combined.argmax(dim=-1).item() == 1
    assert processor.calls == 1
    assert torch.equal(base.seen_input_ids[0], input_ids)
    assert torch.equal(assistant.seen_input_ids[0], input_ids)

    second_combined = processor(torch.tensor([[7, 8, 9]]), base_logits)
    assert torch.equal(second_combined, expected)
    assert processor.calls == 2
    assert torch.equal(assistant.seen_input_ids[1], torch.tensor([[9]]))
