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
    """Test that length mismatches are handled gracefully."""
    import logging
    
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    
    # Create a simple datum
    tokens = tokenizer.encode("Hello world", add_special_tokens=False)
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
    
    # Create mock teacher client that returns wrong length
    teacher_client = MagicMock()
    # Return logprobs with wrong length (too short)
    teacher_logprobs = [0.0, 0.1, 0.2]  # Only 3 values, but should be more
    teacher_client.compute_logprobs_async = AsyncMock(return_value=teacher_logprobs)
    
    # Capture log messages
    log_records = []
    
    def log_handler(record):
        log_records.append(record)
    
    # Add handler to logger
    logger = logging.getLogger("tinker_cookbook.distillation.train_on_policy")
    handler = logging.Handler()
    handler.emit = log_handler
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    
    try:
        # This should not crash, but should log a warning and use uniform multiplier
        metrics = await incorporate_kl_penalty(
            data_D=[datum],
            teacher_clients_D=[teacher_client],
            dataset_indices_D=[0],
            kl_penalty_coef=1.0,
            kl_discount_factor=0.0,
            tokenizer=tokenizer,
            sample_start_index=0,
        )
        
        # Should still return metrics (graceful degradation)
        assert "teacher_kl" in metrics
        
        # Should have logged a warning about length mismatch
        assert len(log_records) > 0
        assert any("mismatch" in record.getMessage().lower() for record in log_records), \
            "Should log a warning about length mismatch"
    finally:
        logger.removeHandler(handler)


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
