import random
from collections import deque

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class MomentumScorePredictor:
    def __init__(self, window_size=10):
        if window_size < 2:
            raise ValueError("Window size must be at least 2")
        self.scores = deque(maxlen=window_size)
        self.delta_window_size = window_size - 1
        self.deltas = deque(maxlen=self.delta_window_size)
        self.last_score = None

    def add_score(self, score):
        self.scores.append(score)
        if len(self.scores) > 1:
            delta = score - self.scores[-2]
            self.deltas.append(delta)
        self.last_score = score

    def predict_next_score(self):
        if len(self.deltas) < self.delta_window_size:
            return None
        average_delta = sum(self.deltas) / len(self.deltas)
        return self.last_score + average_delta


class EWMAScorePredictor:
    def __init__(self, alpha=0.2, initial_score=None):
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        self.ewma = initial_score

    def add_score(self, score):
        if self.ewma is None:
            self.ewma = score
        else:
            self.ewma = self.alpha * score + (1 - self.alpha) * self.ewma

    def predict_next_score(self):
        return self.ewma


class MeanScorePredictor:
    def __init__(self, window_size=100):
        self.scores = deque(maxlen=window_size)

    def add_score(self, score):
        self.scores.append(score)

    def predict_next_score(self):
        if len(self.scores) == 0:
            return None
        return sum(self.scores) / len(self.scores)

    def clear_before(self):
        if len(self.scores) == 0:
            return
        score = self.scores[-1]
        self.scores.clear()
        self.scores.append(score)


def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def initialize_tree(input_ids, model, past_key_values, logits_processor):
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # Clone the output hidden states
    eagle_device = next(model.eagle_layer.parameters()).device
    if outputs["hidden_states"][0].device != eagle_device:
        outputs["hidden_states"] = [
            x.to(eagle_device) for x in outputs["hidden_states"]
        ]
    hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = (
        model.eagle_layer.topK_genrate(hidden_states, input_ids, logits_processor)
    )
    return (
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        orig,
        hidden_states,
        token,
    )


def reset_tree_mode(
    model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    # import pdb; pdb.set_trace()
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    eagle_device = next(model.eagle_layer.parameters()).device
    if outputs["hidden_states"][0].device != eagle_device:
        outputs["hidden_states"] = [
            x.to(eagle_device) for x in outputs["hidden_states"]
        ]
    hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided
    logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or
    evaluates posterior probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor):
        Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float):
        Softmax temperature for probability scaling. A value of 0 indicates
        greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - sample_token (torch.Tensor): Target model recover token of the best candidate.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position
        # in the sequence
        posterior_mask = (
            candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        sample_p = logits[best_candidate, accept_length]
        sample_token = torch.argmax(sample_p)
        sample_token = sample_token[None, None]
        return best_candidate, accept_length, sample_token

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        sample_token = torch.multinomial(sample_p, 1)
        sample_token = sample_token[None]
        return torch.tensor(best_candidate), accept_length - 1, sample_token


@torch.no_grad()
def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    new_token,
    past_key_values_data_list,
    current_length_data,
    model,
    hidden_state_new,
    sample_token,
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [
            input_ids,
            candidates[None, best_candidate, : accept_length + 1].to(input_ids.device),
        ],
        dim=-1,
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based
    # on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[
            ..., select_indices.to(past_key_values_data.device), :
        ]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[
            ..., prev_input_len : prev_input_len + tgt.shape[-2], :
        ]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[
        :, best_candidate, : accept_length + 1
    ]

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, early_stop_signal = (
        model.eagle_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((input_ids, sample_token.to(input_ids.device)), dim=1),
            logits_processor=logits_processor,
        )
    )

    new_token += accept_length + 1

    return (
        input_ids,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        new_token,
        early_stop_signal,
    )
