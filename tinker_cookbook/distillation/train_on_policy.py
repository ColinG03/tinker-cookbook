"""
Implements on-policy distillation. For more details, see:
https://thinkingmachines.ai/blog/on-policy-distillation
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Sequence, cast

import chz
import tinker
import torch

from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import scope, update_scope_context, trace_init

logger = logging.getLogger(__name__)


def check_think_tag_format(
    model_input: tinker.ModelInput,
    tokenizer: Tokenizer,
) -> bool:
    """
    Check whether think tags in the token sequence are well-formed.
    Returns True when every <think> has a matching </think> (and at least one
    pair exists). Returns False when tags are missing entirely or unbalanced.
    """
    tokens = model_input.to_ints()
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)

    n_open = 0
    n_close = 0
    i = 0
    while i < len(tokens):
        if (
            i + len(think_start_tokens) <= len(tokens)
            and tokens[i : i + len(think_start_tokens)] == think_start_tokens
        ):
            n_open += 1
            i += len(think_start_tokens)
        elif (
            i + len(think_end_tokens) <= len(tokens)
            and tokens[i : i + len(think_end_tokens)] == think_end_tokens
        ):
            n_close += 1
            i += len(think_end_tokens)
        else:
            i += 1

    return n_open > 0 and n_open == n_close


def identify_reasoning_tokens(
    model_input: tinker.ModelInput,
    tokenizer: Tokenizer,
) -> torch.Tensor:
    """
    Identify tokens that are part of reasoning blocks, from the start of <think> through
    the end of </think> inclusive. When no closing </think> is found, marks from <think>
    through the end of the sequence.
    
    Args:
        model_input: The ModelInput containing the sequence
        tokenizer: Tokenizer to encode the markers
        
    Returns:
        A boolean tensor of shape (seq_len,) where True indicates reasoning tokens
    """
    tokens = model_input.to_ints()
    
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    
    reasoning_mask = torch.zeros(len(tokens), dtype=torch.bool)
    
    i = 0
    while i < len(tokens):
        if i + len(think_start_tokens) <= len(tokens):
            if tokens[i:i + len(think_start_tokens)] == think_start_tokens:
                block_start = i
                j = i + len(think_start_tokens)
                found_end = False
                while j < len(tokens):
                    if j + len(think_end_tokens) <= len(tokens):
                        if tokens[j:j + len(think_end_tokens)] == think_end_tokens:
                            reasoning_mask[block_start:j + len(think_end_tokens)] = True
                            i = j + len(think_end_tokens)
                            found_end = True
                            break
                    j += 1
                
                if not found_end:
                    reasoning_mask[block_start:] = True
                    break
            else:
                i += 1
        else:
            i += 1
    
    return reasoning_mask


@scope
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    teacher_clients_D: List[tinker.SamplingClient],
    dataset_indices_D: List[int],
    kl_penalty_coef: float,
    kl_discount_factor: float,
    reasoning_kl_multiplier: float,
    tokenizer: Tokenizer,
    sample_start_index: int = 0,
    format_penalty: float = 0.0,
) -> Dict[str, float]:
    """
    Compute reverse KL between the student (log p) and the teacher model (log q), computed as
    log p - log q. We then adjust the advantages in-place as the negative reverse KL.
    
    Reasoning tokens (from <think> through </think>, inclusive of the tags) have their
    KL penalty scaled by reasoning_kl_multiplier (1.0 = full penalty, 0.0 = no penalty).

    Args:
        data_D: List of datums to compute KL for
        teacher_clients_D: List of teacher sampling clients, one per datum
        dataset_indices_D: List of dataset indices, one per datum
        kl_penalty_coef: Coefficient for KL penalty
        kl_discount_factor: Discount factor for future KL
        reasoning_kl_multiplier: Multiplier for KL penalty on think tokens (1.0 = no reduction)
        tokenizer: Tokenizer to identify reasoning tokens
        sample_start_index: Starting index for sample counting
        format_penalty: Penalty applied to advantages when think tags are malformed/missing
    """
    # Note: if your teacher has a different renderer than the student, you may want to modify
    #       the full_sequence_inputs_D to match the teacher's renderer.
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    # Compute the teacher's logprobs for each element of the batch
    # Each datum uses its corresponding teacher sampling client
    teacher_logprobs_D = await asyncio.gather(
        *[
            teacher_client.compute_logprobs_async(sequence_input)
            for teacher_client, sequence_input in zip(teacher_clients_D, full_sequence_inputs_D)
        ]
    )
    # The reverse KL is computed as KL[p||q] = log p - log q, where
    #   - p: sampled_logprobs
    #   - q: teacher_logprobs
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    reverse_kl = [
        (sampled_logprobs - torch.tensor(teacher_logprobs[1:])) * mask
        for teacher_logprobs, sampled_logprobs, mask in safezip(
            teacher_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]
    # Track per-dataset KL for logging
    # dataset_idx -> (sum of KL, sum of mask)
    per_dataset_kl: Dict[int | str, tuple[float, float]] = {}

    for i, datum in enumerate(data_D):
        sample_index = sample_start_index + i
        
        # Identify reasoning tokens (between <think> and </think>) from the full sequence
        # full_sequence_inputs_D[i] contains input + last target token
        reasoning_mask_full = identify_reasoning_tokens(full_sequence_inputs_D[i], tokenizer)
        
        reasoning_multiplier = reasoning_kl_multiplier
        
        # The reverse_kl is computed for target tokens (teacher_logprobs[1:])
        # So we need to extract the reasoning mask for target positions (indices 1 onwards)
        seq_len = len(reverse_kl[i])
        
        # Validate sequence length relationships with graceful fallback
        expected_full_length = len(full_sequence_inputs_D[i].to_ints())
        actual_mask_length = len(reasoning_mask_full)
        expected_target_length = expected_full_length - 1
        
        # Check if reasoning mask length matches full sequence length
        if actual_mask_length != expected_full_length:
            logger.warning(
                f"Reasoning mask length mismatch for datum {i} (sample {sample_index}): "
                f"expected {expected_full_length} (from full_sequence), "
                f"got {actual_mask_length} (from identify_reasoning_tokens). "
                f"Skipping reasoning token scaling for this datum."
            )
            # Fallback: use uniform multiplier (no reasoning token scaling)
            kl_multiplier = torch.ones(seq_len)
        # Check if reverse_kl length matches expected target length
        elif seq_len != expected_target_length:
            logger.warning(
                f"Reverse KL length mismatch for datum {i} (sample {sample_index}): "
                f"expected {expected_target_length} (full_sequence - 1), "
                f"got {seq_len} (from reverse_kl). "
                f"Skipping reasoning token scaling for this datum."
            )
            # Fallback: use uniform multiplier (no reasoning token scaling)
            kl_multiplier = torch.ones(seq_len)
        else:
            # Extract reasoning mask for target positions
            reasoning_mask_target = reasoning_mask_full[1:seq_len + 1]
            
            # Final check: verify the slice worked correctly
            if len(reasoning_mask_target) != seq_len:
                logger.warning(
                    f"Reasoning mask target length mismatch for datum {i} (sample {sample_index}): "
                    f"expected {seq_len}, got {len(reasoning_mask_target)}. "
                    f"reasoning_mask_full length: {len(reasoning_mask_full)}, "
                    f"seq_len: {seq_len}. "
                    f"Skipping reasoning token scaling for this datum."
                )
                # Fallback: use uniform multiplier (no reasoning token scaling)
                kl_multiplier = torch.ones(seq_len)
            else:
                # All checks passed - apply reasoning token scaling
                kl_multiplier = torch.ones(seq_len)
                kl_multiplier[reasoning_mask_target] = reasoning_multiplier
        
        # The advantage is the negative reverse KL. We can optionally apply a discount factor.
        # Apply the multiplier to the KL penalty coefficient for reasoning tokens
        kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i] * kl_multiplier
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

        # Format penalty: penalize malformed/missing think tags
        if format_penalty > 0:
            tags_ok = check_think_tag_format(full_sequence_inputs_D[i], tokenizer)
            if not tags_ok:
                format_advantages = -format_penalty * float_masks[i]
                datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
                    datum.loss_fn_inputs["advantages"].to_torch() + format_advantages
                )
            if "__format_violations" not in per_dataset_kl:
                per_dataset_kl["__format_violations"] = (0.0, 0.0)
            prev_v, prev_t = per_dataset_kl["__format_violations"]
            per_dataset_kl["__format_violations"] = (
                prev_v + (0.0 if tags_ok else 1.0),
                prev_t + 1.0,
            )

        # Accumulate per-dataset KL
        dataset_idx = dataset_indices_D[i]
        kl_sum = reverse_kl[i].sum().item()
        mask_sum = float_masks[i].sum().item()
        if dataset_idx not in per_dataset_kl:
            per_dataset_kl[dataset_idx] = (0.0, 0.0)
        prev_kl_sum, prev_mask_sum = per_dataset_kl[dataset_idx]
        per_dataset_kl[dataset_idx] = (prev_kl_sum + kl_sum, prev_mask_sum + mask_sum)

    # Compute average reverse KL over the batch for logging purposes
    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / sum(
        [mask.sum() for mask in float_masks]
    )

    # Compute per-dataset metrics
    metrics = {"teacher_kl": float(avg_logp_diff)}
    for dataset_idx, (kl_sum, mask_sum) in per_dataset_kl.items():
        if mask_sum > 0:
            metrics[f"teacher_kl/dataset_{dataset_idx}"] = float(kl_sum / mask_sum)

    if "__format_violations" in per_dataset_kl:
        v, t = per_dataset_kl.pop("__format_violations")
        metrics["think_tag_violation_rate"] = float(v / t) if t > 0 else 0.0

    return metrics


@chz.chz
class Config:
    learning_rate: float
    dataset_configs: List[DistillationDatasetConfig]
    model_name: str
    max_tokens: int
    temperature: float = 1.0
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    reasoning_kl_multiplier: float = 1.0
    format_penalty: float = 0.0

    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
    kl_penalty_coef: float,
    kl_discount_factor: float,
    reasoning_kl_multiplier: float,
    sample_start_index: int = 0,
    format_penalty: float = 0.0,
) -> tuple[list[tinker.Datum], dict[str, Any], int]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Print one datum per dataset
    printed_datasets = set()
    for datum, metadata in zip(data_D, metadata_D):
        dataset_idx = dataset_indices_P[metadata["group_idx"]]
        if dataset_idx not in printed_datasets:
            logger.info(colorize_example(datum, tokenizer, key="mask"))
            printed_datasets.add(dataset_idx)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("compute_kl_penalty", metrics):
            # Map each datum to its teacher sampling client and dataset index using metadata
            #   - metadata_D contains group_idx which indexes into trajectory_groups_P
            #   - dataset_indices_P[group_idx] gives us the dataset index
            #   - teacher_clients[dataset_idx] gives us the teacher
            teacher_clients_D = [
                teacher_clients[dataset_indices_P[metadata["group_idx"]]] for metadata in metadata_D
            ]
            dataset_indices_D = [
                dataset_indices_P[metadata["group_idx"]] for metadata in metadata_D
            ]
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                teacher_clients_D,
                dataset_indices_D,
                kl_penalty_coef,
                kl_discount_factor,
                reasoning_kl_multiplier,
                tokenizer,
                sample_start_index,
                format_penalty=format_penalty,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics, len(data_D)


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
    sample_start_index: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any], int]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics, num_samples = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        dataset_indices_P,
        teacher_clients,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        reasoning_kl_multiplier=cfg.reasoning_kl_multiplier,
        sample_start_index=sample_start_index,
        format_penalty=cfg.format_penalty,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D=data_D,
            training_client=training_client,
            learning_rate=cfg.learning_rate,
            num_substeps=cfg.num_substeps,
            loss_fn=cfg.loss_fn,
            loss_fn_config=cfg.loss_fn_config,
            metrics=metrics,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics, num_samples


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: CompositeDataset,
    teacher_clients: List[tinker.SamplingClient],
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    # Track total number of samples processed across all batches
    total_samples_processed = 0

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch and sample trajectories
        env_group_builders_P, dataset_indices_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=False,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )
        trajectory_groups_P = [
            trajectory_group
            for trajectory_group in trajectory_groups_P
            if trajectory_group is not None
        ]

        # Train step
        sampling_client, train_step_metrics, num_samples = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            dataset_indices_P,
            teacher_clients,
            sample_start_index=total_samples_processed,
        )
        total_samples_processed += num_samples

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for on-policy distillation."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        future = await training_client.load_state_with_optimizer_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create datasets and teacher sampling clients from configs
    datasets = []
    teacher_clients = []
    groups_per_batch_list = []
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]

    for dataset_config in cfg.dataset_configs:
        # Create dataset
        dataset, maybe_test_dataset = await dataset_config.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dataset_config.groups_per_batch)

        # Add test dataset evaluator if present
        if maybe_test_dataset is not None:
            evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

        # Create teacher sampling client
        teacher_config = dataset_config.teacher_config
        teacher_client = service_client.create_sampling_client(base_model=teacher_config.base_model)
        # Load teacher checkpoint if specified
        if teacher_config.load_checkpoint_path is not None:
            teacher_client = service_client.create_sampling_client(
                base_model=teacher_config.base_model,
                model_path=teacher_config.load_checkpoint_path,
            )
        teacher_clients.append(teacher_client)
        logger.info(
            f"Created teacher sampling client for {teacher_config.base_model} "
            f"(checkpoint: {teacher_config.load_checkpoint_path})"
        )

    # Wrap datasets in CompositeDataset
    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=composite_dataset,
        teacher_clients=teacher_clients,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
