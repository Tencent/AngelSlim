import os
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_glmasr import GlmasrConfig
from .modeling_audio import WhisperSpecialEncoder

prune_method = os.environ.get('method')
remain_token_ratio = float(os.environ.get('remain_token_ratio', 1))
threshold = float(os.environ.get('threshold', 1))

class AudioMLPAdapter(nn.Module):
    def __init__(self, config: GlmasrConfig):
        super().__init__()
        whisper_config = config.whisper_config
        self.merge_factor = config.merge_factor
        self.whisper = WhisperSpecialEncoder(
            whisper_config,
            use_rope=config.use_rope,
        )
        self.whisper.layer_norm = nn.Identity()
        self.layer_norm = nn.LayerNorm(whisper_config.hidden_size)
        act = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
        }[config.mlp_adapter_act]
        hidden = whisper_config.hidden_size * self.merge_factor
        output_dim = config.lm_config.hidden_size
        self.adapting = nn.Sequential(
            nn.Linear(hidden, output_dim * 2),
            act,
            nn.Linear(output_dim * 2, output_dim),
        )
        self.audio_bos_eos_token = nn.Embedding(2, output_dim)

    def forward(self, audios: Tensor, output_attentions=False) -> tuple[Tensor, Tensor, Tensor]:
        bsz = audios.size(0)
        outputs, attn_key = self.whisper(audios, output_attentions=output_attentions)
        encoded=outputs[0]
        all_attentions=outputs[1]
        encoded = self.layer_norm(encoded)
        encoded = encoded.reshape(bsz, -1, encoded.size(-1) * self.merge_factor)
        adapted = self.adapting(encoded)
        boa = self.audio_bos_eos_token.weight[0][None, :]
        eoa = self.audio_bos_eos_token.weight[1][None, :]
        if output_attentions:
            return adapted, boa, eoa, all_attentions, attn_key
        return adapted, boa, eoa,


class GlmasrModel(LlamaForCausalLM):
    config_class = GlmasrConfig

    def __init__(self, config: GlmasrConfig):
        super().__init__(config.lm_config)
        self.audio_encoder = AudioMLPAdapter(config)
        self.all_config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audios: Optional[Tensor] = None,
        audio_offsets: Optional[list[list[int]]] = None,
        audio_length: Optional[list[list[int]]] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        tokens = input_ids
        vocab_size = self.config.vocab_size
        tokens = torch.clamp(tokens, 0, vocab_size - 1)
        language_embs = self.model.embed_tokens(tokens)

        have_audio = audios is not None and (
            kwargs.get("past_key_values") is None or len(kwargs["past_key_values"]) == 0
        )
        if have_audio:
            if audio_length is None:
                raise ValueError("audio_length is required when audio_offsets are provided")
            audio_embs, boa, eoa, all_attentions, attn_key = self.audio_encoder(audios, output_attentions=True)
            idx = 0
            attn_logits = all_attentions[-1]

            if prune_method == 'visionzip':
                contextual_num_ratio = 0.05
                dominant_num_ratio = remain_token_ratio - contextual_num_ratio
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1

                for i, audio_output_length in enumerate(audio_length[0]):
                    dominant_num = max(int(dominant_num_ratio * audio_output_length),1)
                    contextual_num =  max(int(contextual_num_ratio * audio_output_length),1)

                    attn_logits_i = attn_logits[i].max(dim=0).values # (N)
                    attn_logits_i = attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1).mean(0)
                    attn_logits_i = attn_logits_i[:audio_output_length]
                    topk_indices = torch.topk(attn_logits_i, dominant_num)[1]

                    mask = torch.zeros_like(index_masks[0], dtype=torch.bool)
                    mask[topk_indices] = True
                    contextual_mask = (~mask)
                    contextual_mask[audio_output_length:] = False

                    attn_key = attn_key.transpose(1,2)
                    attn_key = attn_key.mean(dim=1).unsqueeze(1) # [1, 1, 1500, 64]
                    attn_key = attn_key.view(attn_key.shape[0], attn_key.shape[1], attn_key.shape[2]//4, 4, attn_key.shape[-1]).mean(dim=3) # [1, 1, 375, 64]
                    metric_filtered = attn_key[i, :, contextual_mask] # [1, contextual_len, 64]
                    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) # [1, contextual_len, 64]
                    del metric_filtered
                    step = max(1, metric_normalized.shape[1] // contextual_num)
                    target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
                    target_tokens = metric_normalized[:, target_indices, :]
                    tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
                    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
                    assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=attn_logits.dtype, device=metric_normalized.device)
                    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
                    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                    
                    select_mask = torch.zeros_like(index_masks[0], dtype=torch.bool)
                    select_mask[topk_indices] = True
                    false_pos = (~select_mask).nonzero(as_tuple=True)[0]   
                    select_mask[false_pos[target_indices]] = True
                    index_masks[i] = select_mask
                    contextual_mask = contextual_mask.to(audio_embs.device)

                    hidden_states_filtered = audio_embs[i, contextual_mask].unsqueeze(0) # [contextual_len, 4096]
                    target_indices = target_indices.to(hidden_states_filtered.device)
                    hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
                    assign_one_hot = assign_one_hot.to(hidden_to_merge.device)
                    counts = counts.to(hidden_to_merge.device)
                    try:
                        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts # [1, 7, 4096]
                    except:
                        aggregated_hidden = torch.bmm(assign_one_hot.unsqueeze(0).transpose(1, 2), hidden_to_merge) / counts # [1, 7, 4096]
                    target_hidden = hidden_states_filtered[:, target_indices, :]
                    contextual_tokens = target_hidden + aggregated_hidden
                    audio_embs[i, false_pos[target_indices]] = contextual_tokens.squeeze(0)

                audio_features_mask = torch.arange(N, device=device)[None,:].expand_as(index_masks)
                audio_features_mask = audio_features_mask < torch.tensor(audio_length[0], device=device)[:,None]
                audio_embs = audio_embs[audio_features_mask & index_masks]

                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1
                del attn_key
                torch.cuda.empty_cache()
            
            elif prune_method=='vispruner':
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1

                for index, audio_output_length in enumerate(audio_length[0]):
                    remain_token_num = remain_token_num_list[index]
                    important_ratio = 0.5
                    important_token_num = int(torch.round(remain_token_num * important_ratio)) # T_imp = T * r
                    diverse_token_num = remain_token_num - important_token_num # T_div = T * (1 - r)
                    
                    attn_logits_i = attn_logits[index].max(dim=0).values # (N)
                    attn_logits_i = attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1).mean(0)
                    token_indices = attn_logits_i[:audio_output_length].argsort(dim=-1, descending=True) # (N)
                    important_indices = token_indices[:important_token_num] # (T_imp)
                    residual_indices,_ = token_indices[important_token_num:].sort() # (N - T_imp)
                    audio_normalized = audio_embs[index] / audio_embs[index].norm(dim=-1, keepdim=True) # (N, C)
                    while diverse_token_num > 0:
                        R = residual_indices.shape[0]
                        r = int(min(4, R - diverse_token_num))
                        if r <= 0:
                            break
                        residual_tokens = audio_normalized[residual_indices] # (R, C)
                        a, b = residual_tokens[::2, :], residual_tokens[1::2, :] # (R // 2, C)
                        scores = a @ b.transpose(-1, -2) # (R // 2, R // 2)
                        scores = scores.max(dim=-1).values # (R // 2)
                        distinct_indices = scores.argsort(dim=-1, descending=True)[r:] # (R // 2 - r)
                        residual_indices = torch.cat([
                            residual_indices[::2][distinct_indices], 
                            residual_indices[1::2]
                        ], dim=-1) # (R - r)
                    if diverse_token_num > 0:
                        selected_indices = torch.cat([important_indices, residual_indices], dim=-1) # (T)
                    else:
                        selected_indices = important_indices # (T)
                    index_mask = torch.zeros(N, dtype=torch.bool, device=device)
                    index_mask[selected_indices] = True
                    index_masks[index] = index_mask

                audio_features_mask = torch.arange(N, device=device)[None,:].expand_as(index_masks)
                audio_features_mask = audio_features_mask < torch.tensor(audio_length[0], device=device)[:,None]
                audio_embs = audio_embs[audio_features_mask & index_masks]

                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1

            elif prune_method == 'cdpruner':
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1
                
                for index, audio_output_length in enumerate(audio_length[0]):
                    remain_token_num = remain_token_num_list[index]
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                    similarity = torch.matmul(audio_normalized, audio_normalized.transpose(0, 1))
                    kernel = similarity
                    cis = torch.zeros((remain_token_num, audio_output_length), device=device)
                    di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                    select_idx = torch.empty(remain_token_num, dtype=torch.long, device=device)
                    for i in range(remain_token_num):
                        k = torch.argmax(di2s, dim=-1)
                        select_idx[i] = k
                        eis = (kernel[k] - torch.einsum('t,tn->n', cis[:i, k], cis[:i])) / torch.sqrt(di2s[k])
                        cis[i, :] = eis
                        di2s -= torch.square(eis)
                        di2s[k] = -float('inf')
                    select_mask = torch.zeros(N, dtype=torch.bool, device=device)
                    select_mask[select_idx] = True
                    index_masks[index] = select_mask
                audio_features_mask = torch.arange(N, device=device)[None,:].expand_as(index_masks)
                audio_features_mask = audio_features_mask < torch.tensor(audio_length[0], device=device)[:,None]
                audio_embs = audio_embs[audio_features_mask & index_masks]

                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1
            
            elif prune_method == "cdpruner-attention":
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1

                for index, audio_output_length in enumerate(audio_length[0]):
                    remain_token_num = remain_token_num_list[index]
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                    similarity = torch.matmul(audio_normalized, audio_normalized.transpose(0, 1))

                    attn_logits_i = attn_logits[index].max(dim=0).values # (N)
                    attn_logits_i = torch.diagonal(attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1))
                    relevance = attn_logits_i[:audio_output_length]
                    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
                    cis = torch.zeros((remain_token_num, audio_output_length), device=device)
                    di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                    select_idx = torch.empty(remain_token_num, dtype=torch.long, device=device)
                    for i in range(remain_token_num):
                        k = torch.argmax(di2s, dim=-1)
                        select_idx[i] = k
                        eis = (kernel[k] - torch.einsum('t,tn->n', cis[:i, k], cis[:i])) / torch.sqrt(di2s[k])
                        cis[i, :] = eis
                        di2s -= torch.square(eis)
                        di2s[k] = -float('inf')
                    select_mask = torch.zeros(N, dtype=torch.bool, device=device)
                    select_mask[select_idx] = True
                    index_masks[index] = select_mask
                
                audio_features_mask = torch.arange(N, device=device)[None,:].expand_as(index_masks)
                audio_features_mask = audio_features_mask < torch.tensor(audio_length[0], device=device)[:,None]
                audio_embs = audio_embs[audio_features_mask & index_masks]
                    
                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1
            
            elif prune_method == "atome-merge":
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1
                new_audio_features = []
                for index, audio_output_length in enumerate(audio_length[0]):
                    audio_token_num = remain_token_num_list[index]
                    merge_token_num =  audio_output_length - audio_token_num
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                    cos_similarities = F.cosine_similarity(audio_normalized[::2,:][:audio_normalized[1::2,:].shape[0]], audio_normalized[1::2,:], dim=1) # (seq_len - 1)
                    merge_token_num = min(merge_token_num, cos_similarities.shape[0])
                    merge_indexes = torch.topk(cos_similarities, merge_token_num, dim=-1).indices * 2
                    merge_indexes = merge_indexes.sort().values

                    h = []
                    idx = 0
                    while idx < audio_output_length:
                        if idx in merge_indexes:
                            h.append((audio_embs[index][idx] + audio_embs[index][idx + 1])/2.0)
                            idx += 2
                        else:
                            h.append(audio_embs[index][idx])
                            idx += 1
                    new_audio_features.append(torch.stack(h))
                audio_embs = torch.cat(new_audio_features)

                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1
            
            elif prune_method == "atome-merge-attention":
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1

                new_audio_features = []
                for index, audio_output_length in enumerate(audio_length[0]):
                    attn_logits_i = attn_logits[index].max(dim=0).values # (N)
                    attn_logits_i_mask = torch.arange(N, device=device)[None, :]
                    attn_logits_i_mask = attn_logits_i_mask < audio_output_length
                    attn_logits_i = attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1) # 375 * 375
                    attn_logits_i = attn_logits_i * attn_logits_i_mask.transpose(0,1).expand_as(attn_logits_i)
                    attn_logits_i = attn_logits_i.mean(0)
                    importance = attn_logits_i[:audio_output_length]
                    
                    audio_token_num = remain_token_num_list[index] # T
                    merge_token_num =  audio_output_length - audio_token_num
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                                        
                    cos_similarities = F.cosine_similarity(audio_normalized[::2,:][:audio_normalized[1::2,:].shape[0]], audio_normalized[1::2,:], dim=1) # (seq_len - 1)
                    merge_token_num = min(merge_token_num, cos_similarities.shape[0])
                    merge_indexes = torch.topk(cos_similarities, merge_token_num, dim=-1).indices * 2
                    merge_indexes = merge_indexes.sort().values
                    h = []
                    idx = 0
                    while idx < audio_output_length:
                        if idx in merge_indexes:
                            merged_indexes = [idx, idx + 1]
                            h.append((importance[merged_indexes].unsqueeze(-1).expand_as(audio_embs[index][merged_indexes]) * audio_embs[index][merged_indexes]).sum(0) / importance[merged_indexes].sum())
                            idx += 2
                        else:
                            h.append(audio_embs[index][idx])
                            idx += 1
                    new_audio_features.append(torch.stack(h))
                audio_embs = torch.cat(new_audio_features)

                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1

            elif prune_method == "fastadasp":
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1
                new_audio_features = []
                for index, audio_output_length in enumerate(audio_length[0]):
                    audio_token_num = remain_token_num_list[index]
                    merge_token_num =  audio_output_length - audio_token_num
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                    cos_similarities = F.cosine_similarity(audio_normalized[:-1,:], audio_normalized[1:,:], dim=1) # (seq_len - 1)
                    selected_indices = torch.topk(cos_similarities, merge_token_num, dim=-1).indices
                    selected_indices = selected_indices.sort().values
                    h = []
                    idx = 0
                    while idx < audio_output_length:
                        merged_indexes = []
                        while idx in selected_indices:
                            merged_indexes.append(idx)
                            idx += 1
                        merged_indexes.append(idx)
                        h.append(audio_embs[index][merged_indexes].mean(0))
                        idx+=1
                    new_audio_features.append(torch.stack(h))
                    assert new_audio_features[index].shape[0] == audio_token_num

                audio_embs = torch.cat(new_audio_features)
                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1

            elif prune_method == "fastadasp-attention":
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1
                new_audio_features = []
                for index, audio_output_length in enumerate(audio_length[0]):
                    attn_logits_i = attn_logits[index].max(dim=0).values # (N)
                    attn_logits_i_mask = torch.arange(N, device=device)[None, :]
                    attn_logits_i_mask = attn_logits_i_mask < audio_output_length
                    attn_logits_i = attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1) # 750 * 750
                    attn_logits_i = attn_logits_i * attn_logits_i_mask.transpose(0,1).expand_as(attn_logits_i)
                    attn_logits_i = attn_logits_i.mean(0)
                    importance = attn_logits_i[:audio_output_length]

                    audio_token_num = remain_token_num_list[index]
                    merge_token_num =  audio_output_length - audio_token_num
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                    cos_similarities = F.cosine_similarity(audio_normalized[:-1,:], audio_normalized[1:,:], dim=1) # (seq_len - 1)
                    selected_indices = torch.topk(cos_similarities, merge_token_num, dim=-1).indices
                    selected_indices = selected_indices.sort().values
                    h = []
                    idx = 0
                    while idx < audio_output_length:
                        merged_indexes = []
                        while idx in selected_indices:
                            merged_indexes.append(idx)
                            idx += 1
                        merged_indexes.append(idx)
                        h.append((importance[merged_indexes].unsqueeze(-1).expand_as(audio_embs[index][merged_indexes]) * audio_embs[index][merged_indexes]).sum(0) / importance[merged_indexes].sum())
                        idx+=1
                    new_audio_features.append(torch.stack(h))
                    assert new_audio_features[index].shape[0] == audio_token_num

                audio_embs = torch.cat(new_audio_features)
                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1

            elif prune_method == "atome-merge-attention-dpp":
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1
                new_audio_features = []
                for index, audio_output_length in enumerate(audio_length[0]):
                    audio_token_num = remain_token_num_list[index] # T
                    merge_token_num =  int(audio_output_length - audio_token_num)
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                    cos_similarities = F.cosine_similarity(audio_normalized[::2,:][:audio_normalized[1::2,:].shape[0]], audio_normalized[1::2,:], dim=1) # (seq_len - 1)
                    
                    cos_similarities_th = threshold
                    merge_token_num = min((cos_similarities > cos_similarities_th).sum().item(), merge_token_num)
                    merge_indexes = torch.topk(cos_similarities, merge_token_num, dim=-1).indices * 2
                    merge_indexes = merge_indexes.sort().values

                    attn_logits_i = attn_logits[index].max(dim=0).values # (N)
                    attn_logits_i_mask = torch.arange(N, device=device)[None, :]
                    attn_logits_i_mask = attn_logits_i_mask < audio_output_length
                    attn_logits_i = attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1) # 750 * 750
                    attn_logits_i = attn_logits_i * attn_logits_i_mask.transpose(0,1).expand_as(attn_logits_i)
                    merge_attn_logits_i = attn_logits_i.mean(0)
                    merge_importance = merge_attn_logits_i[:audio_output_length]

                    attn_logits_i = attn_logits[index].mean(dim=0) # (N)
                    prune_attn_logits_i = torch.diagonal(attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1))
                    prune_importance = []
                    h = []
                    idx = 0
                    while idx < audio_output_length:
                        if idx in merge_indexes:
                            merged_indexes = [idx, idx + 1]
                            h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_embs[index][merged_indexes]) * audio_embs[index][merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                            prune_importance.append(prune_attn_logits_i[merged_indexes].mean())
                            idx += 2
                        else:
                            h.append(audio_embs[index][idx])
                            prune_importance.append(prune_attn_logits_i[idx])
                            idx += 1

                    new_audio_features.append(torch.stack(h))
                    relevance = torch.stack(prune_importance)

                    new_audio_features_normalized = new_audio_features[index] / new_audio_features[index].norm(dim=-1, keepdim=True) # (N, C)
                    similarity = torch.matmul(new_audio_features_normalized, new_audio_features_normalized.transpose(0, 1))
                    cur_audio_output_length = new_audio_features[index].shape[0]
                    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
                    
                    cis = torch.zeros((audio_token_num, cur_audio_output_length), device=device)
                    di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                    select_idx = torch.empty(audio_token_num, dtype=torch.long, device=device)
                    for i in range(audio_token_num):
                        k = torch.argmax(di2s, dim=-1)
                        select_idx[i] = k
                        eis = (kernel[k] - torch.einsum('t,tn->n', cis[:i, k], cis[:i])) / torch.sqrt(di2s[k])
                        cis[i, :] = eis
                        di2s -= torch.square(eis)
                        di2s[k] = -float('inf')
                    new_audio_features[index]=new_audio_features[index][select_idx.sort()[0]]
                    assert new_audio_features[index].shape[0] == audio_token_num
                audio_embs = torch.cat(new_audio_features)

                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1

            elif prune_method == "fastadasp-attention-dpp":
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1
                new_audio_features = []
                for index, audio_output_length in enumerate(audio_length[0]):
                    audio_token_num = remain_token_num_list[index]
                    merge_token_num =  int(audio_output_length - audio_token_num)
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                    cos_similarities = F.cosine_similarity(audio_normalized[:-1,:], audio_normalized[1:,:], dim=1) # (seq_len - 1)

                    cos_similarities_th = threshold
                    merge_token_num = min((cos_similarities > cos_similarities_th).sum().item(), merge_token_num)
                    selected_indices = torch.topk(cos_similarities, merge_token_num, dim=-1).indices
                    selected_indices = selected_indices.sort().values

                    attn_logits_i = attn_logits[index].max(dim=0).values # (N)
                    attn_logits_i_mask = torch.arange(N, device=device)[None, :]
                    attn_logits_i_mask = attn_logits_i_mask < audio_output_length
                    attn_logits_i = attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1) # 750 * 750
                    attn_logits_i = attn_logits_i * attn_logits_i_mask.transpose(0,1).expand_as(attn_logits_i)
                    merge_attn_logits_i = attn_logits_i.mean(0)
                    merge_importance = merge_attn_logits_i[:audio_output_length]

                    attn_logits_i = attn_logits[index].mean(dim=0) # (N)
                    prune_attn_logits_i = torch.diagonal(attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1))
                    prune_importance = []
                    h = []
                    idx = 0
                    while idx < audio_output_length:
                        merged_indexes = []
                        while idx in selected_indices:
                            merged_indexes.append(idx)
                            idx += 1
                        merged_indexes.append(idx)
                        h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_embs[index][merged_indexes]) * audio_embs[index][merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                        prune_importance.append(prune_attn_logits_i[merged_indexes].mean())
                        idx+=1
                    new_audio_features.append(torch.stack(h))
                    relevance = torch.stack(prune_importance)

                    # dpp-pruner
                    new_audio_features_normalized = new_audio_features[index] / new_audio_features[index].norm(dim=-1, keepdim=True) # (N, C)
                    similarity = torch.matmul(new_audio_features_normalized, new_audio_features_normalized.transpose(0, 1))
                    cur_audio_output_length = new_audio_features[index].shape[0]
                    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)

                    # [CDPruner] Fast MAP inference of conditional DPP
                    cis = torch.zeros((audio_token_num, cur_audio_output_length), device=device)
                    di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                    select_idx = torch.empty(audio_token_num, dtype=torch.long, device=device)
                    for i in range(audio_token_num):
                        k = torch.argmax(di2s, dim=-1)
                        select_idx[i] = k
                        eis = (kernel[k] - torch.einsum('t,tn->n', cis[:i, k], cis[:i])) / torch.sqrt(di2s[k])
                        cis[i, :] = eis
                        di2s -= torch.square(eis)
                        di2s[k] = -float('inf')
                    new_audio_features[index]=new_audio_features[index][select_idx.sort()[0]]
                    assert new_audio_features[index].shape[0] == audio_token_num
                audio_embs = torch.cat(new_audio_features)

                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1
 
            elif prune_method == "SA-MAP":
                B, N, C = audio_embs.shape
                device = audio_embs.device
                index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
                pos_begin = torch.where(input_ids[0] == 59261)[0]
                pos_end = torch.where(input_ids[0] == 59262)[0]
                remain_token_num_list = pos_end-pos_begin - 1
                new_audio_features = []
                for index, audio_output_length in enumerate(audio_length[0]):
                    audio_token_num = remain_token_num_list[index]
                    audio_normalized = audio_embs[index][:audio_output_length] / audio_embs[index][:audio_output_length].norm(dim=-1, keepdim=True) # (N, C)
                    similarity = torch.matmul(audio_normalized, audio_normalized.transpose(0, 1))
                    
                    attn_logits_i = attn_logits[index].max(dim=0).values # (N)
                    attn_logits_i_mask = torch.arange(N, device=device)[None, :]
                    attn_logits_i_mask = attn_logits_i_mask < audio_output_length
                    attn_logits_i = attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1) # 750 * 750
                    attn_logits_i = attn_logits_i * attn_logits_i_mask.transpose(0,1).expand_as(attn_logits_i)
                    merge_attn_logits_i = attn_logits_i.mean(0)
                    merge_importance = merge_attn_logits_i[:audio_output_length]

                    attn_logits_i = attn_logits[index].mean(dim=0) # (N)
                    prune_attn_logits_i = torch.diagonal(attn_logits_i.reshape(N,4,N,4).mean(1).sum(-1))

                    prune_importance = []
                    idx = 0
                    h = []
                    merged_indexes = []
                    while idx < audio_output_length:
                        merged_indexes.append(idx)
                        while idx+1 < audio_output_length:
                            if similarity[merged_indexes][:,idx+1].mean() >= threshold:
                                merged_indexes.append(idx+1)
                                idx+=1
                            else:
                                h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_embs[index][merged_indexes]) * audio_embs[index][merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                                prune_importance.append(prune_attn_logits_i[merged_indexes].mean())
                                merged_indexes = []
                                break
                        idx+=1
                    if merged_indexes != []:
                        h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_embs[index][merged_indexes]) * audio_embs[index][merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                        prune_importance.append(prune_attn_logits_i[merged_indexes].mean())

                    new_audio_features.append(torch.stack(h[1:-1]))
                    prune_importance = torch.stack(prune_importance[1:-1])
                    relevance = prune_importance

                    # dpp-pruner
                    new_audio_features_normalized = new_audio_features[index] / new_audio_features[index].norm(dim=-1, keepdim=True) # (N, C)
                    similarity = torch.matmul(new_audio_features_normalized, new_audio_features_normalized.transpose(0, 1))
                    cur_audio_output_length = new_audio_features[index].shape[0]
                    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
                    cis = torch.zeros((audio_token_num, cur_audio_output_length), device=device)
                    di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                    select_idx = torch.empty(audio_token_num, dtype=torch.long, device=device)
                    for i in range(audio_token_num):
                        k = torch.argmax(di2s, dim=-1)
                        select_idx[i] = k
                        eis = (kernel[k] - torch.einsum('t,tn->n', cis[:i, k], cis[:i])) / torch.sqrt(di2s[k])
                        cis[i, :] = eis
                        di2s -= torch.square(eis)
                        di2s[k] = -float('inf')
                    new_audio_features[index]=new_audio_features[index][select_idx.sort()[0]]
                    assert new_audio_features[index].shape[0] == audio_token_num
                audio_embs = torch.cat(new_audio_features)

                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, remain_token_num_list)):
                    audio_embs_offset = 0
                    for offset, length in zip(offsets, [lengths]):
                        language_embs[batch, offset : offset + length] = audio_embs[audio_embs_offset:length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        audio_embs_offset+=length
                        idx += 1

            else:
                for batch, (offsets, lengths) in enumerate(zip(audio_offsets, audio_length)):
                    for offset, length in zip(offsets, lengths):
                        language_embs[batch, offset : offset + length] = audio_embs[idx, :length]
                        language_embs[batch, offset - 1] = boa
                        language_embs[batch, offset + length] = eoa
                        idx += 1

        kwargs.pop("inputs_embeds", None)
        kwargs.pop("is_first_forward", None)

        outputs = self.model(
            inputs_embeds=language_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.lm_head(outputs[0])
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _update_model_kwargs_for_generation(self, *args, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(*args, **kwargs)
        model_kwargs["is_first_forward"] = False
        position_ids = model_kwargs.get("position_ids")
        if position_ids is not None:
            next_pos = position_ids[..., -1:].clone() + 1
            model_kwargs["position_ids"] = torch.cat([position_ids, next_pos], dim=-1)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        *args,
        past_key_values: Optional[tuple] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        is_first_forward: bool = True,
        **kwargs,
    ):
        prepared = super().prepare_inputs_for_generation(
            *args,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            is_first_forward=is_first_forward,
            **kwargs,
        )
        for key, value in kwargs.items():
            if key not in prepared and key.startswith("audio"):
                prepared[key] = value
        if is_first_forward and past_key_values is not None and len(past_key_values) > 0:
            cached_len = past_key_values[0][0].shape[2]
            prepared["input_ids"] = prepared["input_ids"][:, cached_len:]
            if "position_ids" in prepared:
                prepared["position_ids"] = prepared["position_ids"][:, cached_len:]
        if not is_first_forward:
            prepared["audios"] = None
        return prepared


__all__ = ["GlmasrModel"]
