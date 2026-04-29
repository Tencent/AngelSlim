import yaml
import logging
import time

import os
import torch

from .flow_matching.ode_wrapper import StreamingODEWrapperForPrefix
from .flow_matching.model import DiTPrefix
from .flow_matching.scheduler import StreamingFlowMatchingScheduler


logger = logging.getLogger(__name__)


class StreamingSemanticFMWrapper:
    def __init__(
        self,
        speech_model: DiTPrefix,
        max_kv_cache_tokens=900,
        max_prompt_chunk=2,
        use_cfg=True,
        use_cfg_rescale=True,
        cfg_init=1.5,
        cfg_scale=7.5,
        cfg_schedule="linear",
        cfg_token_id=0,
        normalize_mel=False,
        mel_mean=None,
        mel_std=None,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        self.dtype = torch.bfloat16
        self.speech_model = speech_model.to(device).to(self.dtype)
        self.speech_model = self.speech_model.eval()
        self.device = device
        self.normalize_mel = normalize_mel
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        self.use_cfg = use_cfg
        self.use_cfg_rescale = use_cfg_rescale
        self.cfg_init = cfg_init
        self.cfg_scale = cfg_scale
        self.cfg_schedule = cfg_schedule

        self.incremental_state = {}
        self.condition_cache = {"previous_seqlen": 0}

        logger.info(
            f">>> SemanticFMWrapper initialized with use_cfg={use_cfg}, use_cfg_rescale={use_cfg_rescale}, cfg_init={cfg_init}, cfg_scale={cfg_scale}, cfg_schedule={cfg_schedule}"
        )

        self.scheduler = StreamingFlowMatchingScheduler()
        self.ode_wrapper = StreamingODEWrapperForPrefix(
            net=self.speech_model,
            x_mask=None,
            x_cond=None,
            use_cfg=use_cfg,
            use_cfg_rescale=use_cfg_rescale,
            cfg_init=cfg_init,
            cfg_scale=cfg_scale,
            cfg_schedule=cfg_schedule,
            cfg_token_id=cfg_token_id,
        )

        self.max_kv_cache_tokens = max_kv_cache_tokens
        self.max_prompt_chunk = max_prompt_chunk
        self.reserve_kv_cache_tokens = 0

    @torch.inference_mode()
    def infer_chunk(
        self,
        xt_chunk,
        semantic_tokens_chunk,
        start_position_id,
        cache=None,
        look_ahead_tokens=0,
        ode_steps=15,
        verbose=False,
        ode_solver="neural_ode_euler",
    ):
        """
        semantic_tokens: [T_1], torch.LongTensor
        xt: [T_2, 80], torch.Tensor, DO NOT normalize it outside
        ode_steps: int, number of ode steps, default 15
        verbose: bool, default False
        ode_solver: str, ode solver, expected in ("neural_ode_euler", "naive_euler"), default "neural_ode_euler"
        """
        bs = 1

        self.scheduler.set_timesteps(ode_steps)

        semantic_tokens_chunk = semantic_tokens_chunk.unsqueeze(0).to(self.device)
        xt_chunk = xt_chunk.unsqueeze(0).to(self.device).to(self.dtype)

        t_span = torch.linspace(0, 1, self.scheduler.timesteps)

        x_mask = torch.zeros(bs, xt_chunk.shape[1], device=self.device).bool()

        cache_ret = self.ode_wrapper.set_conditions(
            x_mask=x_mask,
            x_cond=semantic_tokens_chunk,
            start_position_id=start_position_id,
            cache=self.condition_cache,
        )

        if verbose:
            t_start = time.time()
        if ode_solver == "neural_ode_euler":
            x_t = self.scheduler.sample_by_neuralode(
                self.ode_wrapper, time_steps=t_span, xt=xt_chunk, verbose=False
            )
        elif ode_solver == "naive_euler":
            x_t = self.scheduler.sample(
                ode_wrapper=self.ode_wrapper,
                time_steps=t_span,
                xt=xt_chunk,
                verbose=False,
            )
        else:
            raise NotImplementedError(
                "ode_solver should be in ('neural_ode_euler', 'naive_euler')"
            )

        if look_ahead_tokens > 0:
            semantic_tokens_left = semantic_tokens_chunk.view(-1)[-look_ahead_tokens:]
            cache["semantic_token"] = semantic_tokens_left
            x_t_ret = x_t[:, :-look_ahead_tokens, :]
        else:
            x_t_ret = x_t

        if look_ahead_tokens > 0:
            x_mask = torch.zeros(
                bs, xt_chunk.shape[1] - look_ahead_tokens, device=self.device
            ).bool()
            self.condition_cache = self.ode_wrapper.set_conditions(
                x_mask=x_mask,
                x_cond=semantic_tokens_chunk[:, :-look_ahead_tokens],
                start_position_id=start_position_id,
                cache=self.condition_cache,
            )
            self.ode_wrapper(torch.Tensor([0.999]).to(x_t_ret.device), x_t_ret)
        else:
            self.condition_cache = cache_ret

        if verbose:
            t_end = time.time()
            logger.info(f"[ODE Chunk] Time cost: {t_end - t_start}")

        if self.normalize_mel:
            x_t_ret = x_t_ret * self.mel_std + self.mel_mean
        return x_t_ret.squeeze(0)

    @torch.inference_mode()
    def infer_mel(
        self,
        semantic_tokens,
        ode_steps=15,
        chunk_size=150,
        verbose=False,
        ode_solver="neural_ode_euler",
    ):
        """
        semantic_tokens: [T_1], torch.LongTensor
        prompt: [T_2, 80], torch.Tensor, DO NOT normalize it outside
        prompt_semantic_tokens, [T_2], torch.LongTensor
        ode_steps: int, number of ode steps, default 15
        verbose: bool, default False
        ode_solver: str, ode solver, expected in ("neural_ode_euler", "naive_euler"), default "neural_ode_euler"
        """
        assert semantic_tokens.dim() == 1

        x_t = torch.randn(semantic_tokens.shape[0], 80).to(self.device).to(self.dtype)

        seq_len = semantic_tokens.shape[0]

        num_chunks = seq_len // chunk_size
        if seq_len % chunk_size != 0:
            num_chunks += 1

        x_pred_collect = []

        if verbose:
            t_start = time.time()

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, seq_len)
            semantic_tokens_chunk = semantic_tokens[start:end]
            x_t_chunk = x_t[start:end, :]

            x_pred = self.infer_chunk(
                xt_chunk=x_t_chunk,
                semantic_tokens_chunk=semantic_tokens_chunk,
                start_position_id=self.start_position_id,
                ode_steps=ode_steps,
                verbose=verbose,
                ode_solver=ode_solver,
            )
            self.start_position_id += end - start
            self.update_incremental_state()

            x_pred_collect.append(x_pred)

        if verbose:
            t_end = time.time()
            logger.info(f"[ODE] Time cost: {t_end - t_start}")

        x_pred = torch.cat(x_pred_collect, dim=0)

        return x_pred

    def clear_all_states(self):
        self.start_position_id = 0
        self.condition_cache = {"previous_seqlen": 0}
        self.ode_wrapper.clear_all_states()

    def state_dict(self):
        return {
            "start_position_id": self.start_position_id,
            "ode_wrapper": self.ode_wrapper.state_dict(),
            "condition_cache": self.condition_cache,
        }

    def load_state_dict(self, state_dict):
        if state_dict is not None:
            self.start_position_id = state_dict["start_position_id"]
            self.ode_wrapper.load_state_dict(state_dict["ode_wrapper"])
            self.condition_cache = state_dict["condition_cache"]

    def update_incremental_state(self):
        self.ode_wrapper.update_incremental_state(
            reserve_kv_cache_tokens=0,
            max_kv_cache_tokens=self.max_kv_cache_tokens,
            condition_cache=self.condition_cache,
        )

    @torch.inference_mode()
    def prefill(self, mel, semantic_token, chunk_size=150, verbose=False):
        """
        mel: [T, 80], torch.Tensor
        semantic_token: [T], torch.LongTensor
        chunk_size: int, default 150
        """
        assert mel.dim() == 2
        assert semantic_token.dim() == 1
        assert (
            semantic_token.shape[0] == mel.shape[0]
        ), "Semantic token and mel shape mismatch"
        seq_len = mel.shape[0]
        num_chunks = min(seq_len // chunk_size, self.max_prompt_chunk)
        start_pos = seq_len - num_chunks * chunk_size

        res_mel = mel[:start_pos, :]
        res_semantic_token = semantic_token[:start_pos]
        self.prefill_chunk(
            res_mel, res_semantic_token, start_position_id=self.start_position_id
        )
        self.start_position_id += start_pos
        self.update_incremental_state()
        self.reserve_kv_cache_tokens += self.ode_wrapper.kv_cache_tokens

        if verbose:
            logger.info("Prefilling prompt with {} chunks".format(num_chunks))
            start_time = time.time()

        for chunk_id in range(num_chunks):
            start = start_pos + chunk_id * chunk_size
            end = start + chunk_size
            mel_chunk = mel[start:end, :]
            semantic_token_chunk = semantic_token[start:end]

            self.prefill_chunk(
                mel_chunk,
                semantic_token_chunk,
                start_position_id=self.start_position_id,
            )
            self.start_position_id += end - start

            self.update_incremental_state()
            self.reserve_kv_cache_tokens += self.ode_wrapper.kv_cache_tokens

        if verbose:
            logger.info(
                "Prefilling done in {:.2f} seconds".format(time.time() - start_time)
            )

    def prefill_chunk(self, mel_chunk, semantic_tokens_chunk, start_position_id=0):
        """
        mel_chunk: [T, 80], torch.Tensor, T is the chunk size
        semantic_tokens_chunk: [T], torch.LongTensor
        start_position_id: int, default 0
        """
        bs = 1

        semantic_tokens_chunk = semantic_tokens_chunk.unsqueeze(0).to(self.device)
        mel_chunk = mel_chunk.unsqueeze(0).to(self.device).to(self.dtype)

        if self.normalize_mel:
            mel_chunk = (mel_chunk - self.mel_mean) / self.mel_std

        x_mask = torch.zeros(bs, mel_chunk.shape[1], device=self.device).bool()

        self.condition_cache = self.ode_wrapper.set_conditions(
            x_mask=x_mask,
            x_cond=semantic_tokens_chunk,
            start_position_id=start_position_id,
            cache=self.condition_cache,
        )

        x_t = torch.Tensor([0.999]).to(self.device)

        self.ode_wrapper(x_t, mel_chunk)

    @classmethod
    def from_pretrained(
        cls,
        model_config,
        ckpt_path,
        device,
        max_prompt_chunk=2,
        max_kv_cache_tokens=900,
        use_cfg=True,
        use_cfg_rescale=True,
        cfg_init=1.5,
        cfg_scale=7.5,
        cfg_schedule="linear",
    ):

        # open yaml file
        with open(model_config, "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]["dit"]
        dit = DiTPrefix(
            input_size=model_config["input_size"],
            semantic_vocab_size=model_config["semantic_vocab_size"] + 1,
            hidden_size=model_config["hidden_size"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            ffn_type=model_config.get("ffn_type", "conv1d_conv1d"),
            ffn_gated_glu=model_config.get("ffn_gated_glu", True),
            ffn_act_layer=model_config.get("ffn_act_layer", "gelu"),
            ffn_conv_kernel_size=model_config.get("ffn_conv_kernel_size", 5),
            use_rope=model_config.get("use_rope", False),
            rope_params=model_config.get(
                "rope_params",
                {
                    "max_position_embeddings": 4096,
                    "rope_base": 10000,
                    "rope_interpolation_factor": 1,
                },
            ),
            position_embedding_type=model_config["position_embedding_type"],
            max_seq_len=model_config["max_seq_len"],
            output_size=model_config["input_size"],
            prompt_cfg_dropout=0,
        )
        cfg_semantic_token_id = model_config["semantic_vocab_size"]

        # load state_dict
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
        speech_model_params = {
            k.replace("speech_model.", ""): v
            for k, v in state_dict.items()
            if "speech_model" in k
        }
        dit.load_state_dict(speech_model_params, strict=True)
        logger.info(f">>> Loaded checkpoint from {ckpt_path}")

        return cls(
            speech_model=dit,
            device=device,
            normalize_mel=config["normalize_mel"],
            mel_mean=config["mel_mean"],
            mel_std=config["mel_std"],
            max_prompt_chunk=max_prompt_chunk,
            max_kv_cache_tokens=max_kv_cache_tokens,
            use_cfg=use_cfg,
            use_cfg_rescale=use_cfg_rescale,
            cfg_init=cfg_init,
            cfg_scale=cfg_scale,
            cfg_schedule=cfg_schedule,
            cfg_token_id=cfg_semantic_token_id,
        )
