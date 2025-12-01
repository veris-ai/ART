import math
import random
from dataclasses import dataclass
from itertools import takewhile
from typing import Generator, cast

import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..trajectories import History, TrajectoryGroup, get_messages


def _mask_assistant_tokens_by_special_tokens(
    token_ids: list[int],
    tokenizer: "PreTrainedTokenizerBase",
) -> list[int]:
    """
    Create assistant mask by finding <|im_start|>assistant...<|im_end|> sections.

    This approach masks all tokens in assistant messages, including both
    content (thinking) and tool_calls, which are formatted by the chat template.

    Args:
        token_ids: Token IDs from apply_chat_template
        tokenizer: Tokenizer to get special token IDs

    Returns:
        Binary mask where 1 = train on this token, 0 = skip
    """
    assistant_mask = [0] * len(token_ids)

    # Get special tokens
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

    # Find all assistant sections
    i = 0
    while i < len(token_ids):
        if token_ids[i] == im_start_id:
            # Check if next tokens are "assistant"
            next_tokens = token_ids[i + 1 : i + 1 + len(assistant_token_ids)]
            if next_tokens == assistant_token_ids:
                # Found assistant section start
                start = i
                # Find corresponding <|im_end|>
                end_idx = i + 1
                while end_idx < len(token_ids) and token_ids[end_idx] != im_end_id:
                    end_idx += 1

                if end_idx < len(token_ids):
                    # Mask from <|im_start|> to <|im_end|> inclusive
                    end = end_idx + 1
                    assistant_mask[start:end] = [1] * (end - start)
                    i = end
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    return assistant_mask


@dataclass
class TokenizedResult:
    advantage: float
    chat: str
    tokens: list[str]
    token_ids: list[int]
    input_pos: list[int]
    assistant_mask: list[int]
    logprobs: list[float]
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    weight: float = 0.0
    prompt_id: int = 0
    prompt_length: int = 0

    def without_prompt(self) -> "TokenizedResult":
        return TokenizedResult(
            advantage=self.advantage,
            chat=self.chat,
            tokens=self.tokens[self.prompt_length :],
            token_ids=self.token_ids[self.prompt_length :],
            input_pos=self.input_pos[self.prompt_length :],
            assistant_mask=self.assistant_mask[self.prompt_length :],
            logprobs=self.logprobs[self.prompt_length :],
            pixel_values=None,
            image_grid_thw=None,
            weight=self.weight,
            prompt_id=self.prompt_id,
            prompt_length=0,
        )


def tokenize_trajectory_groups(
    tokenizer: "PreTrainedTokenizerBase",
    trajectory_groups: list[TrajectoryGroup],
    allow_training_without_logprobs: bool,
    scale_rewards: bool,
    shuffle_group_trajectories: bool = True,
    image_processor: BaseImageProcessor | None = None,
) -> Generator["TokenizedResult", None, None]:
    for group_idx, group in enumerate(trajectory_groups):
        if not group:
            continue
        results: list[TokenizedResult] = []
        # Calculate GRPO group mean and standard deviation
        reward_mean = sum(trajectory.reward for trajectory in group) / len(group)
        reward_std = math.sqrt(
            sum((trajectory.reward - reward_mean) ** 2 for trajectory in group)
            / len(group)
        )
        for traj_idx, trajectory in enumerate(group):
            # Calculate GRPO advantage for this trajectory
            advantage = trajectory.reward - reward_mean
            if scale_rewards:
                advantage /= reward_std + 1e-6
            # Skip trajectories with no advantage
            if advantage == 0:
                continue
            trajectory_results: list[TokenizedResult] = []
            for history in [
                History(
                    messages_and_choices=trajectory.messages_and_choices,
                    tools=trajectory.tools,
                ),
                *trajectory.additional_histories,
            ]:
                if result := tokenize_trajectory(
                    tokenizer,
                    image_processor,
                    history,
                    advantage,
                    allow_training_without_logprobs,
                ):
                    trajectory_results.append(result)
            weight = 1 / (
                sum(sum(result.assistant_mask) for result in trajectory_results) + 1e-6
            )
            for result in trajectory_results:
                result.weight = weight
            results.extend(trajectory_results)
        # Choose a random prompt id
        prompt_id = random.randint(-(2**63), 2**63 - 1)
        # Find the longest shared prefix
        # TODO: Potentially support multiple prompts per group
        # Initial thought is to sort the results by token_ids and then
        # successively group prompts with the same prefix.
        prompt_length = len(
            list(
                takewhile(
                    lambda x: len(set(x)) == 1,
                    zip(*(r.token_ids for r in results)),
                )
            )
        )
        first_non_nan_index = min(
            (
                next(
                    (i for i, lp in enumerate(r.logprobs) if not math.isnan(lp)),
                    len(r.logprobs),
                )
                for r in results
            ),
            default=0,
        )
        prompt_length = max(min(prompt_length, first_non_nan_index) - 1, 0)
        # Set the prompt id and length
        for result in results:
            result.prompt_id = prompt_id
            result.prompt_length = prompt_length

        # Log group-level stats
        import logging
        logger = logging.getLogger(__name__)
        if results:
            total_group_tokens = sum(len(r.token_ids) for r in results)
            total_group_masked = sum(sum(r.assistant_mask) for r in results)
            logger.info(
                f"Group {group_idx}: {len(results)} trajectories, "
                f"{total_group_tokens} total tokens, "
                f"{total_group_masked} masked ({100*total_group_masked/total_group_tokens:.1f}%)"
            )

        if shuffle_group_trajectories:
            random.shuffle(results)
        yield from results


def tokenize_trajectory(
    tokenizer: "PreTrainedTokenizerBase",
    image_processor: BaseImageProcessor | None,
    history: History,
    advantage: float,
    allow_training_without_logprobs: bool,
) -> TokenizedResult | None:
    """
    Tokenizes a trajectory and returns a TokenizedResult.
    """
    # Find the index of the last assistant message
    last_assistant_index = -1
    for i, message in enumerate(history.messages_and_choices):
        if isinstance(message, dict):
            role = message.get("role")
            has_logprobs = "logprobs" in message
            logprobs_value = message.get("logprobs") if has_logprobs else None
            is_assistant = role == "assistant"
            condition_result = is_assistant and (logprobs_value or allow_training_without_logprobs)
            if condition_result:
                last_assistant_index = i
        elif not isinstance(message, dict):
            has_logprobs = bool(message.logprobs) if hasattr(message, "logprobs") else False
            condition_result = has_logprobs or allow_training_without_logprobs
            if condition_result:
                last_assistant_index = i
    # If there are no trainable assistant messages, return None
    if last_assistant_index == -1:
        return None
    messages_and_choices = history.messages_and_choices[: last_assistant_index + 1]
    messages = get_messages(messages_and_choices)
    tools = (
        [{"type": "function", "function": tool} for tool in history.tools]
        if history.tools is not None
        else None
    )
    try:
        chat = cast(
            str,
            tokenizer.apply_chat_template(
                cast(list[dict], messages),
                tools=tools,  # type: ignore
                continue_final_message=True,
                tokenize=False,
            ),
        )
    except ValueError as e:
        # Debug: Print the problematic messages
        import json
        print(f"\n{'='*80}")
        print(f"ERROR in apply_chat_template with continue_final_message=True")
        print(f"Error: {e}")
        print(f"Number of messages: {len(messages)}")
        print(f"Last message: {json.dumps(messages[-1], indent=2)}")
        if len(messages) >= 2:
            print(f"Second-to-last message: {json.dumps(messages[-2], indent=2)}")
        print(f"All messages:")
        for i, msg in enumerate(messages):
            print(f"  [{i}] role={msg.get('role')}, content_length={len(str(msg.get('content', '')))}, has_tool_calls={'tool_calls' in msg}")
        print(f"{'='*80}\n")
        raise
    try:
        original_token_ids = cast(
            list[int],
            tokenizer.apply_chat_template(
                cast(list[dict], messages),
                tools=tools,  # type: ignore
                continue_final_message=True,
            ),
        )
    except ValueError as e:
        # This shouldn't fail if the first call succeeded, but just in case
        print(f"\nERROR: Second apply_chat_template call failed: {e}")
        raise

    # Use special-token-based masking (assumes logprobs will be recomputed via precalculate)
    token_ids = original_token_ids
    assistant_mask = _mask_assistant_tokens_by_special_tokens(token_ids, tokenizer)
    logprobs = [float("nan")] * len(token_ids)

    # Log masking stats for verification
    total_tokens = len(token_ids)
    masked_tokens = sum(assistant_mask)
    tool_call_token_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    tool_call_positions = [i for i, tid in enumerate(token_ids) if tid == tool_call_token_id]

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"Tokenized trajectory: {total_tokens} tokens, "
        f"{masked_tokens} masked ({100*masked_tokens/total_tokens:.1f}%), "
        f"{len(tool_call_positions)} tool calls found"
    )

    if tool_call_positions:
        # Verify tool calls are masked
        for pos in tool_call_positions:
            if assistant_mask[pos] == 1:
                logger.info(f"  ✓ Tool call at position {pos} is masked (mask=1)")
            else:
                logger.warning(f"  ✗ Tool call at position {pos} NOT masked (mask=0) - BUG!")
    else:
        logger.debug("  No tool calls in this trajectory")
    if image_processor:
        images: list[Image.Image] = []
        for message in messages_and_choices:
            if (
                isinstance(message, dict)
                and message["role"] == "user"
                and isinstance(message["content"], (list, tuple))
            ):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        image_url = content["image_url"]["url"].removeprefix("file://")
                        images.append(Image.open(image_url))
        image_token_id = cast(
            int,
            getattr(image_processor, "image_token_id", None)
            or tokenizer.convert_tokens_to_ids(  # type: ignore
                getattr(image_processor, "image_token", "<|image_pad|>")
            ),
        )
        if images:
            result = image_processor(images=images)
            offset = 0
            for num_image_tokens in (
                image_grid_thw.prod().item()
                // (getattr(image_processor, "merge_size", 1) ** 2)
                for image_grid_thw in result["image_grid_thw"]
            ):
                start = token_ids.index(image_token_id, offset)
                offset = start + num_image_tokens
                end = start + 1
                token_ids[start:end] = [image_token_id] * num_image_tokens
                logprobs[start:end] = [float("nan")] * num_image_tokens
                assistant_mask[start:end] = [0] * num_image_tokens
            pixel_values = result["pixel_values"]
            image_grid_thw = result["image_grid_thw"]
        else:
            pixel_values = None
            image_grid_thw = None
    else:
        pixel_values = None
        image_grid_thw = None
    return TokenizedResult(
        advantage=advantage,
        chat=chat,
        tokens=[tokenizer.decode(token_id) for token_id in token_ids],
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=assistant_mask,
        logprobs=logprobs,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
