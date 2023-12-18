from vllm.model_executor.layers.sampler import (
    _prune_hidden_states, 
    _apply_top_p_top_k,
    _apply_penalties,
    _apply_logits_processors,
    _apply_min_p,
    _get_logprobs,
    _build_sampler_output,
    _get_logits,
    _sample
)
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sequence import SamplerOutput

from typing import Optional
import torch
from torch import nn

class IlqlSampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence, frequency and repetition penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self._copy_stream: torch.cuda.Stream = torch.cuda.Stream()

    def forward(
        self,
        embedding: torch.Tensor,
        logit_bias: torch.Tensor, # beta * advantage term from π(a|h) ∝ πβ (a|h)exp[β(Q(h,a)−V (h))]
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, embedding, embedding_bias,
                             self.vocab_size)

        _, vocab_size = logits.shape

        # Apply logits processors (if any).
        logits = _apply_logits_processors(logits, sampling_metadata)

        # Prepare sampling tensors in another stream to overlap
        # CPU<->GPU data transfer with GPU computation in forward pass.
        with torch.cuda.stream(self._copy_stream):
            (sampling_tensors, do_penalties, do_top_p_top_k,
             do_min_p) = SamplingTensors.from_sampling_metadata(
                 sampling_metadata, vocab_size, logits.device, logits.dtype)

        torch.cuda.current_stream().wait_stream(self._copy_stream)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)

        # Apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_p_top_k(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)


        # compute pseudo-logits with logit_bias (from ilql) added
        mask = logits > -float("inf")
        logits = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        logits.add_(logit_bias * mask)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        sample_results = _sample(probs, logprobs, sampling_metadata)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results, sampling_metadata,
                                     prompt_logprobs, sample_logprobs)