import torch
import torch.nn as nn
from functools import lru_cache
import copy


@lru_cache(maxsize=1)
def get_cached_zeros(numel, device="cpu", dtype=torch.float32):
    return torch.zeros(numel, device=device, dtype=dtype)


class StreamingODEWrapperForPrefix(nn.Module):
    def __init__(
        self,
        net,
        x_mask,
        x_cond,
        use_cfg=False,
        use_cfg_rescale=True,
        cfg_init=1.0,
        cfg_scale=4.0,
        cfg_schedule="linear",
        cfg_token_id=0,
    ):
        super(StreamingODEWrapperForPrefix, self).__init__()
        self.net = net
        self.x_mask = x_mask
        self.x_cond = x_cond

        assert use_cfg == False, "cfg is not supported in streaming detokenizer"

        self.use_cfg = use_cfg
        self.use_cfg_rescale = use_cfg_rescale
        self.cfg_init = cfg_init
        self.cfg_scale = cfg_scale
        self.cfg_token_id = cfg_token_id
        self.cfg_schedule = cfg_schedule
        self.position_ids = None
        self.seq_len = None

        self.incremental_state = {}
        self.kv_cache_tokens = 0
        self.cu_seqlens = None
        self.cu_maxlen = None

        self.cu_seqlens_k = None
        self.cu_maxlen_k = None
        self.previous_seqlen = None

    def clear_all_states(self):
        self.incremental_state = {}
        self.kv_cache_tokens = 0
        self.cu_seqlens = None
        self.cu_maxlen = None

        self.cu_seqlens_k = None
        self.cu_maxlen_k = None
        self.previous_seqlen = None

    def state_dict(self):
        return {
            "incremental_state": copy.deepcopy(self.incremental_state),
            "kv_cache_tokens": copy.deepcopy(self.kv_cache_tokens),
            "cu_seqlens": copy.deepcopy(self.cu_seqlens),
            "cu_maxlen": copy.deepcopy(self.cu_maxlen),
            "cu_seqlens_k": copy.deepcopy(self.cu_seqlens_k),
            "cu_maxlen_k": copy.deepcopy(self.cu_maxlen_k),
            "previous_seqlen": copy.deepcopy(self.previous_seqlen),
        }

    def load_state_dict(self, state_dict):
        self.incremental_state = state_dict["incremental_state"]
        self.kv_cache_tokens = state_dict["kv_cache_tokens"]
        self.cu_seqlens = state_dict["cu_seqlens"]
        self.cu_maxlen = state_dict["cu_maxlen"]
        self.cu_seqlens_k = state_dict["cu_seqlens_k"]
        self.cu_maxlen_k = state_dict["cu_maxlen_k"]
        self.previous_seqlen = state_dict["previous_seqlen"]

    def set_conditions(self, x_mask, x_cond, start_position_id, cache={}):
        if not self.use_cfg:
            self.x_mask = x_mask
            self.x_cond = x_cond
        else:
            self.x_cond = torch.cat((x_cond, x_cond), dim=0)
            self.x_mask = torch.cat((x_mask, x_mask), dim=0)

        position_ids_cur = [
            i
            for i in range(start_position_id, self.x_cond.shape[1] + start_position_id)
        ]
        position_ids = torch.tensor([position_ids_cur])

        if not self.use_cfg:
            self.position_ids = position_ids.to(self.x_cond.device).long()
            self.seq_len = (
                torch.Tensor([position_ids.shape[1]]).to(self.x_cond.device).long()
            )
        else:
            self.position_ids = (
                torch.cat((position_ids, position_ids), dim=0)
                .to(self.x_cond.device)
                .long()
            )
            self.seq_len = (
                torch.Tensor([position_ids.shape[1], position_ids.shape[1]])
                .to(self.x_cond.device)
                .long()
            )

        cu_seqlens = torch.cumsum(self.seq_len, dim=0)
        self.cu_seqlens = torch.cat(
            [torch.Tensor([0]).to(cu_seqlens.device), cu_seqlens], dim=0
        ).int()
        self.cu_maxlen = self.seq_len.cpu().max()

        if self.cu_seqlens_k is None:
            self.cu_seqlens_k = self.cu_seqlens
            self.cu_maxlen_k = self.cu_maxlen
            previous_seqlen = self.seq_len
        else:
            previous_seqlen_old = cache["previous_seqlen"]
            previous_seqlen = previous_seqlen_old + self.seq_len
            # calculate cu_seqlens_k
            cu_seqlens_k = torch.cumsum(previous_seqlen, dim=0)
            self.cu_seqlens_k = torch.cat(
                [torch.Tensor([0]).to(cu_seqlens_k.device), cu_seqlens_k], dim=0
            ).int()
            self.cu_maxlen_k = previous_seqlen.cpu().max()
        self.previous_seqlen = previous_seqlen
        ret_cache = {"previous_seqlen": previous_seqlen}
        return ret_cache

    def update_incremental_state(
        self,
        reserve_kv_cache_tokens=0,
        max_kv_cache_tokens=900,
        condition_cache={"previous_seqlen"},
    ):

        assert (
            reserve_kv_cache_tokens <= max_kv_cache_tokens
        ), "reserve_kv_cache_tokens should be less than or equal to max_kv_cache_tokens"

        for layer_idx, layer_cache in self.incremental_state.items():
            # update attention kv cache
            layer_cache["attn_kvcache"]["prev_k"] = layer_cache["attn_kvcache"]["cur_k"]
            layer_cache["attn_kvcache"]["prev_v"] = layer_cache["attn_kvcache"]["cur_v"]

            self.kv_cache_tokens = layer_cache["attn_kvcache"]["prev_k"].shape[1]

            if self.kv_cache_tokens > max_kv_cache_tokens:
                # drop old tokens from reserve kv cache tokens to max_kv_cache_tokens
                reserve_tokens_excludeprompt = (
                    max_kv_cache_tokens - reserve_kv_cache_tokens
                )

                if reserve_kv_cache_tokens == 0:
                    layer_cache["attn_kvcache"]["prev_k"] = layer_cache["attn_kvcache"][
                        "prev_k"
                    ][:, -reserve_tokens_excludeprompt:]
                    layer_cache["attn_kvcache"]["prev_v"] = layer_cache["attn_kvcache"][
                        "prev_v"
                    ][:, -reserve_tokens_excludeprompt:]
                elif reserve_tokens_excludeprompt == 0:
                    layer_cache["attn_kvcache"]["prev_k"] = layer_cache["attn_kvcache"][
                        "prev_k"
                    ][:, :reserve_kv_cache_tokens]
                    layer_cache["attn_kvcache"]["prev_v"] = layer_cache["attn_kvcache"][
                        "prev_v"
                    ][:, :reserve_kv_cache_tokens]
                else:
                    layer_cache["attn_kvcache"]["prev_k"] = torch.cat(
                        [
                            layer_cache["attn_kvcache"]["prev_k"][
                                :, :reserve_kv_cache_tokens
                            ],
                            layer_cache["attn_kvcache"]["prev_k"][
                                :, -reserve_tokens_excludeprompt:
                            ],
                        ],
                        dim=1,
                    )

                    layer_cache["attn_kvcache"]["prev_v"] = torch.cat(
                        [
                            layer_cache["attn_kvcache"]["prev_v"][
                                :, :reserve_kv_cache_tokens
                            ],
                            layer_cache["attn_kvcache"]["prev_v"][
                                :, -reserve_tokens_excludeprompt:
                            ],
                        ],
                        dim=1,
                    )

                bsz = layer_cache["attn_kvcache"]["prev_k"].shape[0]
                self.previous_seqlen = (
                    torch.Tensor(
                        [
                            layer_cache["attn_kvcache"]["prev_k"].shape[1]
                            for i in range(bsz)
                        ]
                    )
                    .to(layer_cache["attn_kvcache"]["prev_k"].device)
                    .long()
                )
                condition_cache["previous_seqlen"] = self.previous_seqlen
                self.kv_cache_tokens = layer_cache["attn_kvcache"]["prev_k"].shape[1]

            # clear current cache
            layer_cache["attn_kvcache"].pop("cur_k")
            layer_cache["attn_kvcache"].pop("cur_v")

    def forward(self, t, x, args=None):
        # t = torch.tensor([t * 1000] * x.shape[0], device=x.device, dtype=x.dtype).long()
        t = (
            get_cached_zeros(x.shape[0], device=x.device, dtype=torch.long)
            + (t * 1000).long()
        )

        if self.use_cfg:
            raise NotImplementedError("cfg is not supported in streaming detokenizer.")
        else:
            pred_noise = self.net(
                x=x,
                condition=self.x_cond,
                t=t,
                position_ids=self.position_ids,
                cu_seqlens=self.cu_seqlens,
                cu_maxlen=self.cu_maxlen,
                cu_seqlens_k=self.cu_seqlens_k,
                cu_maxlen_k=self.cu_maxlen_k,
                incremental_state=self.incremental_state,
                nopadding=True,
                mask=None,
                seq_len=None,
            )
            return pred_noise
