"""Tests for distillation training functions."""

import asyncio
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock

import tinker
from tinker.types import EncodedTextChunk

from tinker_cookbook.distillation.train_on_policy import (
    identify_reasoning_tokens,
    incorporate_kl_penalty,
    swap_system_prompt_tokens,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


# =============================================================================
# Tests for identify_reasoning_tokens
# =============================================================================


def test_identify_reasoning_tokens_single_block():
    """Test identifying reasoning tokens in a sequence with one <think> block."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    # Create a sequence: "Hello <think>reasoning</think> world"
    text = "Hello <think>reasoning</think> world"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)
    
    reasoning_mask = identify_reasoning_tokens(model_input, tokenizer)
    
    # Verify the mask has correct length
    assert len(reasoning_mask) == len(tokens)
    
    # Find where <think> and </think> tokens are
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    
    # Find positions
    think_start_idx = None
    think_end_idx = None
    for i in range(len(tokens) - len(think_start_tokens) + 1):
        if tokens[i:i + len(think_start_tokens)] == think_start_tokens:
            think_start_idx = i
            break
    
    for i in range(len(tokens) - len(think_end_tokens) + 1):
        if tokens[i:i + len(think_end_tokens)] == think_end_tokens:
            think_end_idx = i
            break
    
    assert think_start_idx is not None, "Should find <think> marker"
    assert think_end_idx is not None, "Should find </think> marker"
    
    # Verify reasoning mask marks tokens from <think> to </think> inclusive
    assert reasoning_mask[think_start_idx:think_end_idx + len(think_end_tokens)].all(), \
        "All tokens from <think> to </think> should be marked as reasoning"
    
    # Verify tokens before <think> are not reasoning
    if think_start_idx > 0:
        assert not reasoning_mask[:think_start_idx].any(), \
            "Tokens before <think> should not be marked as reasoning"
    
    # Verify tokens after </think> are not reasoning
    end_idx = think_end_idx + len(think_end_tokens)
    if end_idx < len(tokens):
        assert not reasoning_mask[end_idx:].any(), \
            "Tokens after </think> should not be marked as reasoning"


def test_identify_reasoning_tokens_multiple_blocks():
    """Test identifying reasoning tokens with multiple <think> blocks."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    # Create a sequence with multiple thinking blocks
    text = "Start <think>first</think> middle <think>second</think> end"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)
    
    reasoning_mask = identify_reasoning_tokens(model_input, tokenizer)
    
    assert len(reasoning_mask) == len(tokens)
    
    # Count how many reasoning tokens we have
    num_reasoning = reasoning_mask.sum().item()
    
    # Should have at least the marker tokens marked as reasoning
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    min_reasoning_tokens = 2 * (len(think_start_tokens) + len(think_end_tokens))
    
    assert num_reasoning >= min_reasoning_tokens, \
        f"Should have at least {min_reasoning_tokens} reasoning tokens (markers), got {num_reasoning}"


def test_identify_reasoning_tokens_no_thinking():
    """Test that sequences without thinking tokens return all False."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    text = "This is a normal sequence without any thinking tokens"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)
    
    reasoning_mask = identify_reasoning_tokens(model_input, tokenizer)
    
    assert len(reasoning_mask) == len(tokens)
    assert not reasoning_mask.any(), "No reasoning tokens should be found"


def test_identify_reasoning_tokens_unclosed_think():
    """Test handling of unclosed <think> tag."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    # Create a sequence with unclosed <think>
    text = "Start <think>unclosed reasoning"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)
    
    reasoning_mask = identify_reasoning_tokens(model_input, tokenizer)
    
    assert len(reasoning_mask) == len(tokens)
    
    # Should mark everything from <think> to end as reasoning
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_start_idx = None
    for i in range(len(tokens) - len(think_start_tokens) + 1):
        if tokens[i:i + len(think_start_tokens)] == think_start_tokens:
            think_start_idx = i
            break
    
    assert think_start_idx is not None
    assert reasoning_mask[think_start_idx:].all(), \
        "Unclosed <think> should mark everything to end as reasoning"


def test_identify_reasoning_tokens_nested_blocks():
    """Test that nested <think> blocks are handled correctly."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    # Create nested thinking blocks (though this shouldn't happen in practice)
    text = "Start <think>outer <think>inner</think> more outer</think> end"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)
    
    reasoning_mask = identify_reasoning_tokens(model_input, tokenizer)
    
    assert len(reasoning_mask) == len(tokens)
    # Should mark all tokens from first <think> to last </think>
    assert reasoning_mask.any(), "Should find some reasoning tokens"


# =============================================================================
# Tests for incorporate_kl_penalty
# =============================================================================


async def _test_incorporate_kl_penalty_basic_async():
    """Test basic KL penalty incorporation with reasoning token scaling."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    # Create a simple datum with thinking tokens
    text = "Hello <think>reasoning</think> world"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Create model_input (right-shifted, missing last token)
    model_input = tinker.ModelInput.from_ints(tokens[:-1])
    target_tokens = tokens[1:]  # Left-shifted
    
    # Create mock datum
    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": tinker.TensorData.from_torch(torch.randn(len(target_tokens))),
            "mask": tinker.TensorData.from_torch(torch.ones(len(target_tokens))),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(target_tokens))),
        },
    )
    
    # Create mock teacher client
    teacher_client = MagicMock()
    full_sequence = model_input.append_int(target_tokens[-1])
    teacher_logprobs = torch.randn(len(full_sequence.to_ints())).tolist()
    teacher_client.compute_logprobs_async = AsyncMock(return_value=teacher_logprobs)
    
    # Test with sample_index < 500 (should use 0.0 multiplier for reasoning tokens)
    metrics = await incorporate_kl_penalty(
        data_D=[datum],
        teacher_clients_D=[teacher_client],
        dataset_indices_D=[0],
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        reasoning_kl_multiplier=1.0,
        tokenizer=tokenizer,
        sample_start_index=0,  # First sample, so index 0 < 500
    )
    
    # Verify metrics were returned
    assert "teacher_kl" in metrics
    
    # Verify advantages were updated
    updated_advantages = datum.loss_fn_inputs["advantages"].to_torch()
    assert updated_advantages.shape == (len(target_tokens),)


def test_incorporate_kl_penalty_basic():
    """Wrapper to run async test."""
    asyncio.run(_test_incorporate_kl_penalty_basic_async())


async def _test_incorporate_kl_penalty_reasoning_multiplier_async():
    """Test that reasoning token multiplier is applied correctly."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    # Create a datum with thinking tokens
    text = "Hello <think>reasoning</think> world"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    model_input = tinker.ModelInput.from_ints(tokens[:-1])
    target_tokens = tokens[1:]
    
    # Create datum with known logprobs
    sampled_logprobs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens[:len(sampled_logprobs)])),
            "logprobs": tinker.TensorData.from_torch(sampled_logprobs),
            "mask": tinker.TensorData.from_torch(torch.ones(len(sampled_logprobs))),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(sampled_logprobs))),
        },
    )
    
    # Create mock teacher client
    teacher_client = MagicMock()
    full_sequence = model_input.append_int(target_tokens[-1])
    # Create teacher logprobs that differ from sampled
    teacher_logprobs = (sampled_logprobs + 0.1).tolist()
    # Pad with one extra for the full sequence
    teacher_logprobs = [0.0] + teacher_logprobs
    teacher_client.compute_logprobs_async = AsyncMock(return_value=teacher_logprobs)
    
    # Test with sample_index >= 500 (should use 0.3 multiplier for reasoning tokens)
    initial_advantages = datum.loss_fn_inputs["advantages"].to_torch().clone()
    
    await incorporate_kl_penalty(
        data_D=[datum],
        teacher_clients_D=[teacher_client],
        dataset_indices_D=[0],
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        reasoning_kl_multiplier=0.3,
        tokenizer=tokenizer,
        sample_start_index=500,  # >= 500, so should use 0.3 multiplier
    )
    
    # Verify advantages were updated
    updated_advantages = datum.loss_fn_inputs["advantages"].to_torch()
    assert not torch.equal(updated_advantages, initial_advantages), \
        "Advantages should be updated"


def test_incorporate_kl_penalty_reasoning_multiplier():
    """Wrapper to run async test."""
    asyncio.run(_test_incorporate_kl_penalty_reasoning_multiplier_async())


async def _test_incorporate_kl_penalty_length_mismatch_handling_async():
    """Test that teacher logprobs shorter than sampled_logprobs raises a RuntimeError."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")

    # Use a longer sentence so sampled_logprobs has more tokens than the mock teacher returns
    tokens = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog and keeps on running",
        add_special_tokens=False,
    )
    model_input = tinker.ModelInput.from_ints(tokens[:-1])
    target_tokens = tokens[1:]

    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": tinker.TensorData.from_torch(torch.randn(len(target_tokens))),
            "mask": tinker.TensorData.from_torch(torch.ones(len(target_tokens))),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(target_tokens))),
        },
    )

    # Return far fewer logprobs than needed — this is a genuine length mismatch
    teacher_client = MagicMock()
    teacher_client.compute_logprobs_async = AsyncMock(return_value=[0.0, 0.1, 0.2])

    with pytest.raises(RuntimeError):
        await incorporate_kl_penalty(
            data_D=[datum],
            teacher_clients_D=[teacher_client],
            dataset_indices_D=[0],
            kl_penalty_coef=1.0,
            kl_discount_factor=0.0,
            reasoning_kl_multiplier=1.0,
            tokenizer=tokenizer,
            sample_start_index=0,
        )


def test_incorporate_kl_penalty_length_mismatch_handling():
    """Wrapper to run async test."""
    asyncio.run(_test_incorporate_kl_penalty_length_mismatch_handling_async())


async def _test_incorporate_kl_penalty_multiple_datums_async():
    """Test KL penalty with multiple datums."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    # Create multiple datums
    datums = []
    teacher_clients = []
    
    for i in range(3):
        text = f"Example {i} <think>reasoning {i}</think> end"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        model_input = tinker.ModelInput.from_ints(tokens[:-1])
        target_tokens = tokens[1:]
        
        datum = tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": tinker.TensorData.from_torch(torch.randn(len(target_tokens))),
                "mask": tinker.TensorData.from_torch(torch.ones(len(target_tokens))),
                "advantages": tinker.TensorData.from_torch(torch.zeros(len(target_tokens))),
            },
        )
        datums.append(datum)
        
        # Create mock teacher client for this datum
        teacher_client = MagicMock()
        full_sequence = model_input.append_int(target_tokens[-1])
        teacher_logprobs = torch.randn(len(full_sequence.to_ints())).tolist()
        teacher_client.compute_logprobs_async = AsyncMock(return_value=teacher_logprobs)
        teacher_clients.append(teacher_client)
    
    # Test with sample_start_index that puts some samples before 500, some after
    metrics = await incorporate_kl_penalty(
        data_D=datums,
        teacher_clients_D=teacher_clients,
        dataset_indices_D=[0, 0, 0],
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        reasoning_kl_multiplier=1.0,
        tokenizer=tokenizer,
        sample_start_index=499,  # First sample is 499 (< 500), second is 500 (>= 500)
    )
    
    # Verify metrics
    assert "teacher_kl" in metrics
    assert "teacher_kl/dataset_0" in metrics
    
    # Verify all datums had their advantages updated
    for datum in datums:
        updated_advantages = datum.loss_fn_inputs["advantages"].to_torch()
        assert updated_advantages.shape[0] > 0


def test_incorporate_kl_penalty_multiple_datums():
    """Wrapper to run async test."""
    asyncio.run(_test_incorporate_kl_penalty_multiple_datums_async())


# =============================================================================
# Tests for swap_system_prompt_tokens
# =============================================================================


def _build_qwen3_sequence(tokenizer, system_prompt: str, user_msg: str, assistant_msg: str) -> list[int]:
    """Build a Qwen3 chat-template token sequence from raw parts."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)


def test_swap_system_prompt_basic():
    """Swapping produces a sequence with the new system prompt and preserves the rest."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    original_sys = "You are a friendly tutor."
    new_sys = "You are a Socratic tutor who always asks guiding questions."

    tokens = _build_qwen3_sequence(tokenizer, original_sys, "Hi", "Hello!")
    model_input = tinker.ModelInput.from_ints(tokens)

    swapped = swap_system_prompt_tokens(model_input, tokenizer, new_sys)
    swapped_text = tokenizer.decode(swapped.to_ints())

    assert new_sys in swapped_text, "New system prompt should appear in the swapped sequence"
    assert original_sys not in swapped_text, "Original system prompt should be gone"
    # User and assistant turns must survive
    assert "Hi" in swapped_text
    assert "Hello!" in swapped_text


def test_swap_system_prompt_preserves_suffix_exactly():
    """Everything after the system message should be byte-identical."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    original_sys = "Short prompt."
    new_sys = "A much longer and more detailed Socratic system prompt with many extra tokens."

    tokens = _build_qwen3_sequence(tokenizer, original_sys, "What is 2+2?", "Let me think...")
    model_input = tinker.ModelInput.from_ints(tokens)

    # Find where the original system message ends so we can compare the suffix
    im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    system_role_tokens = tokenizer.encode("system\n", add_special_tokens=False)
    im_start_tokens = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    header_len = len(im_start_tokens) + len(system_role_tokens)

    # Locate end of system message in original
    header_tokens = im_start_tokens + system_role_tokens
    sys_start = -1
    for i in range(len(tokens) - len(header_tokens) + 1):
        if tokens[i : i + len(header_tokens)] == header_tokens:
            sys_start = i
            break
    assert sys_start >= 0

    sys_end = -1
    for i in range(sys_start + header_len, len(tokens) - len(im_end_tokens) + 1):
        if tokens[i : i + len(im_end_tokens)] == im_end_tokens:
            sys_end = i + len(im_end_tokens)
            break
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    if tokens[sys_end : sys_end + len(newline_tokens)] == newline_tokens:
        sys_end += len(newline_tokens)

    original_suffix = tokens[sys_end:]

    swapped = swap_system_prompt_tokens(model_input, tokenizer, new_sys)
    swapped_tokens = swapped.to_ints()

    # The suffix should appear unchanged at the end of the swapped sequence
    assert swapped_tokens[-len(original_suffix):] == original_suffix, (
        "Token suffix after the system message must be preserved exactly"
    )


def test_swap_system_prompt_no_system_message_raises():
    """Raise ValueError when the sequence has no system message."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")

    # Build a sequence with only a user turn (no system message)
    plain_text = "Just some plain text without chat template markers"
    tokens = tokenizer.encode(plain_text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)

    with pytest.raises(ValueError, match="Could not find system message start"):
        swap_system_prompt_tokens(model_input, tokenizer, "new prompt")


def test_swap_system_prompt_round_trip():
    """Swapping back to the original prompt recovers the original token sequence."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    original_sys = "You are a helpful assistant."
    tokens = _build_qwen3_sequence(tokenizer, original_sys, "Hello", "Hi there")
    model_input = tinker.ModelInput.from_ints(tokens)

    swapped = swap_system_prompt_tokens(model_input, tokenizer, "Temporary prompt")
    restored = swap_system_prompt_tokens(swapped, tokenizer, original_sys)

    assert restored.to_ints() == tokens, "Round-tripping back to original prompt should recover exact tokens"


# =============================================================================
# Tests for incorporate_kl_penalty with teacher_system_prompt
# =============================================================================


def _make_datum_from_qwen3_chat(tokenizer, system_prompt: str, user_msg: str, assistant_msg: str):
    """Create a (datum, full_sequence_tokens) pair from a Qwen3 chat."""
    tokens = _build_qwen3_sequence(tokenizer, system_prompt, user_msg, assistant_msg)
    model_input = tinker.ModelInput.from_ints(tokens[:-1])
    target_tokens = tokens[1:]
    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": tinker.TensorData.from_torch(torch.randn(len(target_tokens))),
            "mask": tinker.TensorData.from_torch(torch.ones(len(target_tokens))),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(target_tokens))),
        },
    )
    return datum, tokens


async def _test_teacher_system_prompt_modifies_teacher_input_async():
    """When teacher_system_prompt is set, compute_logprobs_async receives a modified input."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    student_sys = "You are a tutor."
    teacher_sys = "You are a Socratic tutor who asks guiding questions rather than giving answers."

    datum, tokens = _make_datum_from_qwen3_chat(tokenizer, student_sys, "What is 2+2?", "Think about it.")
    full_seq = datum.model_input.append_int(tokens[-1])

    teacher_client = MagicMock()
    teacher_logprobs = torch.randn(len(full_seq.to_ints())).tolist()
    teacher_client.compute_logprobs_async = AsyncMock(return_value=teacher_logprobs)

    await incorporate_kl_penalty(
        data_D=[datum],
        teacher_clients_D=[teacher_client],
        dataset_indices_D=[0],
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        reasoning_kl_multiplier=1.0,
        tokenizer=tokenizer,
        teacher_system_prompt=teacher_sys,
    )

    # The teacher should have been called with a modified input
    teacher_client.compute_logprobs_async.assert_called_once()
    called_input = teacher_client.compute_logprobs_async.call_args[0][0]
    called_text = tokenizer.decode(called_input.to_ints())

    assert teacher_sys in called_text, "Teacher should see the overridden system prompt"
    assert student_sys not in called_text, "Teacher should NOT see the student system prompt"


def test_teacher_system_prompt_modifies_teacher_input():
    asyncio.run(_test_teacher_system_prompt_modifies_teacher_input_async())


async def _test_teacher_system_prompt_none_is_backward_compatible_async():
    """When teacher_system_prompt is None, the teacher sees the same input as before."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    student_sys = "You are a tutor."

    datum, tokens = _make_datum_from_qwen3_chat(tokenizer, student_sys, "Hi", "Hello")
    full_seq = datum.model_input.append_int(tokens[-1])

    teacher_client = MagicMock()
    teacher_logprobs = torch.randn(len(full_seq.to_ints())).tolist()
    teacher_client.compute_logprobs_async = AsyncMock(return_value=teacher_logprobs)

    await incorporate_kl_penalty(
        data_D=[datum],
        teacher_clients_D=[teacher_client],
        dataset_indices_D=[0],
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        reasoning_kl_multiplier=1.0,
        tokenizer=tokenizer,
        teacher_system_prompt=None,
    )

    called_input = teacher_client.compute_logprobs_async.call_args[0][0]
    assert called_input.to_ints() == full_seq.to_ints(), (
        "With teacher_system_prompt=None, teacher input must equal the student's full sequence"
    )


def test_teacher_system_prompt_none_is_backward_compatible():
    asyncio.run(_test_teacher_system_prompt_none_is_backward_compatible_async())


async def _test_teacher_system_prompt_does_not_alter_student_data_async():
    """The student's model_input and advantages base are never modified by the teacher override."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    student_sys = "You are a tutor."
    teacher_sys = "You are a very detailed Socratic tutor."

    datum, tokens = _make_datum_from_qwen3_chat(tokenizer, student_sys, "Question?", "Answer.")
    full_seq = datum.model_input.append_int(tokens[-1])

    # Snapshot the student's model_input tokens before the call
    student_tokens_before = datum.model_input.to_ints()

    teacher_client = MagicMock()
    teacher_logprobs = torch.randn(len(full_seq.to_ints())).tolist()
    teacher_client.compute_logprobs_async = AsyncMock(return_value=teacher_logprobs)

    await incorporate_kl_penalty(
        data_D=[datum],
        teacher_clients_D=[teacher_client],
        dataset_indices_D=[0],
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        reasoning_kl_multiplier=1.0,
        tokenizer=tokenizer,
        teacher_system_prompt=teacher_sys,
    )

    assert datum.model_input.to_ints() == student_tokens_before, (
        "Student model_input must not be modified by teacher_system_prompt"
    )


def test_teacher_system_prompt_does_not_alter_student_data():
    asyncio.run(_test_teacher_system_prompt_does_not_alter_student_data_async())


async def _test_teacher_system_prompt_fallback_on_missing_system_msg_async():
    """If the sequence has no system message, falls back gracefully with a warning."""
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")

    # Build a sequence without a system message (just user/assistant)
    messages = [
        {"role": "user", "content": "Hey"},
        {"role": "assistant", "content": "Hi"},
    ]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    model_input = tinker.ModelInput.from_ints(tokens[:-1])
    target_tokens = tokens[1:]

    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": tinker.TensorData.from_torch(torch.randn(len(target_tokens))),
            "mask": tinker.TensorData.from_torch(torch.ones(len(target_tokens))),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(target_tokens))),
        },
    )

    full_seq = model_input.append_int(tokens[-1])
    teacher_client = MagicMock()
    teacher_logprobs = torch.randn(len(full_seq.to_ints())).tolist()
    teacher_client.compute_logprobs_async = AsyncMock(return_value=teacher_logprobs)

    # Should not raise — falls back to original
    metrics = await incorporate_kl_penalty(
        data_D=[datum],
        teacher_clients_D=[teacher_client],
        dataset_indices_D=[0],
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        reasoning_kl_multiplier=1.0,
        tokenizer=tokenizer,
        teacher_system_prompt="This should trigger fallback",
    )

    assert "teacher_kl" in metrics
    # Verify the teacher was called with the original (unmodified) input
    called_input = teacher_client.compute_logprobs_async.call_args[0][0]
    assert called_input.to_ints() == full_seq.to_ints()


def test_teacher_system_prompt_fallback_on_missing_system_msg():
    asyncio.run(_test_teacher_system_prompt_fallback_on_missing_system_msg_async())
