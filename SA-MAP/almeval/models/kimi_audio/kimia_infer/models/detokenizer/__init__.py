import torch
import os
from .bigvgan_wrapper import BigVGANWrapper
from .semantic_fm_prefix_streaming import StreamingSemanticFMWrapper


class PrefixStreamingFlowMatchingDetokenizer:
    def __init__(
        self,
        vocoder: BigVGANWrapper,
        fm: StreamingSemanticFMWrapper,
        look_ahead_tokens: int = 0,
    ) -> None:
        self.dtype = torch.bfloat16

        print("Currently using bfloat16 for PrefixFlowMatchingDetokenizer")

        self.vocoder = vocoder
        self.vocoder.to_dtype(self.dtype)

        self.semantic_fm = fm

        # initialize mel_spec
        self.max_pos_size = 4096
        self.is_timbre_semantic_token = False
        self.pre_mel = None
        self.frame_size = 480  # how many samples in a frame
        self.pre_wav = None
        self.state_dict_backup = None
        self.hamming_window_cache = {}
        self.previous_chunk_left = None
        self.look_ahead_tokens = look_ahead_tokens

        self.clear_states()

    @classmethod
    def from_pretrained(
        cls,
        vocoder_config,
        vocoder_ckpt,
        fm_config,
        fm_ckpt,
        device,
        look_ahead_tokens=0,
        max_prompt_chunk=2,
        max_kv_cache_tokens=900,
        use_cfg=False,
        use_cfg_rescale=True,
        cfg_init=1.5,
        cfg_scale=7.5,
        cfg_schedule="linear",
    ):
        bigvgan = BigVGANWrapper.from_pretrained(vocoder_config, vocoder_ckpt, device)
        semantic_fm = StreamingSemanticFMWrapper.from_pretrained(
            fm_config,
            fm_ckpt,
            device,
            max_prompt_chunk=max_prompt_chunk,
            max_kv_cache_tokens=max_kv_cache_tokens,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
            use_cfg_rescale=use_cfg_rescale,
            cfg_init=cfg_init,
            cfg_schedule=cfg_schedule,
        )
        return cls(bigvgan, semantic_fm, look_ahead_tokens=look_ahead_tokens)

    @torch.inference_mode()
    def prefill(
        self, timbre_speech, timbre_semantic_token, chunk_size: int, timbre_mel=None
    ):
        """
        Arguments:
            timbre_speech: torch.Tensor, shape [B, N_speech_24k]
            timbre_semantic_token: torch.Tensor, shape [B, N]
            chunk_size: int, chunk size for prefilling
            timbre_mel: torch.Tensor, shape [B, N, 80], optional, if not None, use this mel spectrogram instead of extracting from timbre_speech
        """
        if timbre_mel is None:
            assert (
                timbre_speech is not None
            ), "timbre_speech should not be None if timbre_mel is not None"
            assert (
                len(timbre_semantic_token.shape) == 2
                and len(timbre_speech.shape) == 2
                and chunk_size > 0
            )
            assert timbre_speech.shape[0] == 1 and timbre_semantic_token.shape[0] == 1

            mel_spec = self.vocoder.extract_mel_from_wav(
                wav_data=timbre_speech.squeeze(0)
            )
        else:
            assert (
                len(timbre_mel.shape) == 3
                and len(timbre_semantic_token.shape) == 2
                and chunk_size > 0
            )
            assert timbre_mel.shape[0] == 1 and timbre_semantic_token.shape[0] == 1
            mel_spec = timbre_mel.squeeze(0)

        if mel_spec.shape[0] < timbre_semantic_token.shape[1]:
            # pad mel_spec
            mel_spec = torch.nn.functional.pad(
                mel_spec, (0, 0, 0, timbre_semantic_token.shape[1] - mel_spec.shape[0])
            )
        elif mel_spec.shape[0] > timbre_semantic_token.shape[1]:
            # truncate mel_spec
            mel_spec = mel_spec[: timbre_semantic_token.shape[1], :]

        # clear all states
        self.semantic_fm.clear_all_states()
        self.semantic_fm.prefill(
            mel_spec,
            timbre_semantic_token.squeeze(0),
            chunk_size=chunk_size,
            verbose=False,
        )
        self.state_dict_backup = self.semantic_fm.state_dict()

    @torch.inference_mode()
    def detokenize_streaming(
        self,
        semantic_token,
        ode_step=30,
        verbose=False,
        ode_solver="neural_ode_euler",
        is_final=False,
        upsample_factor=1,
    ):
        assert len(semantic_token.shape) == 2 and ode_step > 0
        assert semantic_token.shape[0] == 1

        semantic_token = semantic_token.repeat_interleave(upsample_factor, dim=1)

        semantic_token = semantic_token.squeeze(0)

        if self.look_ahead_tokens != 0 and self.previous_chunk_left is not None:
            semantic_token_previous = self.previous_chunk_left["semantic_token"]
            semantic_token = torch.cat(
                [semantic_token_previous, semantic_token], dim=-1
            )

        x_t_chunk = (
            torch.randn(semantic_token.shape[0], 80)
            .to(semantic_token.device)
            .to(self.dtype)
        )

        if self.look_ahead_tokens != 0 and self.previous_chunk_left is None:
            self.previous_chunk_left = {"semantic_token": None}

        speech_mel = self.semantic_fm.infer_chunk(
            xt_chunk=x_t_chunk,
            semantic_tokens_chunk=semantic_token,
            start_position_id=self.semantic_fm.start_position_id,
            ode_steps=ode_step,
            verbose=verbose,
            look_ahead_tokens=(
                self.look_ahead_tokens * upsample_factor if not is_final else 0
            ),
            cache=self.previous_chunk_left,
            ode_solver=ode_solver,
        )

        chunk_size = speech_mel.shape[0]
        length = speech_mel.shape[0]
        self.semantic_fm.start_position_id += length
        self.semantic_fm.update_incremental_state()
        self.semantic_fm.reserve_kv_cache_tokens += (
            self.semantic_fm.ode_wrapper.kv_cache_tokens
        )

        # smoothing

        # I will maintain the history of seqlen wav
        # For the first chunk, I will only return the half chunk wav, and save the res half chunk in history
        # For the rest requests, I will concat the generated wav with the history, output one chunk of the history, save the

        if self.pre_mel is None:  # first chunk, related to TTFB
            concat_mel = speech_mel
            concat_reconstructed_wav = self.vocoder.decode_mel(concat_mel)
            if is_final:
                self.clear_states()
                self.state_dict_backup = None
                ret_wav = concat_reconstructed_wav.float()
            else:
                reconstructed_wav = concat_reconstructed_wav[
                    :, : int(self.frame_size * chunk_size // 2)
                ]  # return the first half chunk

                self.pre_wav = concat_reconstructed_wav[
                    :, -int(self.frame_size * chunk_size // 2) :
                ]  # log the last half chunk for next generation step
                self.pre_mel = speech_mel[-chunk_size // 2 :, :]

                ret_wav = reconstructed_wav.float()
        else:
            concat_mel = torch.cat([self.pre_mel, speech_mel], dim=0)
            concat_reconstructed_wav = self.vocoder.decode_mel(concat_mel)

            if is_final:
                self.clear_states()
                self.state_dict_backup = None
                ret_wav = concat_reconstructed_wav.float()
            else:
                # fetch history
                prev_speech_len = self.pre_wav.shape[1]

                if concat_reconstructed_wav.shape[1] > prev_speech_len * 2:
                    gen_speech_len = prev_speech_len * 2
                else:
                    gen_speech_len = concat_reconstructed_wav.shape[1] // 2

                reconstructed_wav = concat_reconstructed_wav[
                    :, :gen_speech_len
                ]  # return the first half chunk

                if gen_speech_len not in self.hamming_window_cache:
                    self.hamming_window_cache[gen_speech_len] = (
                        torch.hamming_window(gen_speech_len)
                        .to(self.dtype)
                        .to(semantic_token.device)
                        .unsqueeze(0)
                    )

                hamming_window = self.hamming_window_cache[gen_speech_len]

                # apply smoothing of the first half chunk
                reconstructed_wav[:, : int(gen_speech_len // 2)] = (
                    self.pre_wav[:, : int(gen_speech_len // 2)]
                    * hamming_window[:, -int(gen_speech_len // 2) :]
                    + reconstructed_wav[:, : int(gen_speech_len // 2)]
                    * hamming_window[:, : int(gen_speech_len // 2)]
                )

                res_speech_len = concat_reconstructed_wav.shape[1] - gen_speech_len
                res_mel_len = res_speech_len // self.frame_size

                self.pre_wav = concat_reconstructed_wav[:, -res_speech_len:]
                self.pre_mel = speech_mel[-res_mel_len:, :]
                ret_wav = reconstructed_wav.float()

        if (
            not is_final
            and self.semantic_fm.start_position_id + 2 * chunk_size > self.max_pos_size
        ):
            # out of position id,
            self.semantic_fm.clear_all_states()
            self.semantic_fm.load_state_dict(self.state_dict_backup)

        return ret_wav

    def clear_states(self):
        self.semantic_fm.clear_all_states()
        self.previous_chunk_left = None
        self.pre_mel = None
        self.pre_wav = None


def get_audio_detokenizer(model_path):
    fm_model_config = os.path.join(model_path, "audio_detokenizer", "config.yaml")
    fm_ckpt_path = os.path.join(model_path, "audio_detokenizer", "model.pt")

    bigvgan_config_file = os.path.join(model_path, "vocoder", "config.json")
    bigvgan_ckpt_path = os.path.join(model_path, "vocoder", "model.pt")

    device = torch.cuda.current_device()
    detokenizer = PrefixStreamingFlowMatchingDetokenizer.from_pretrained(
        vocoder_config=bigvgan_config_file,
        vocoder_ckpt=bigvgan_ckpt_path,
        max_prompt_chunk=10,  # 10 * 3 = 30s
        fm_config=fm_model_config,
        fm_ckpt=fm_ckpt_path,
        device=device,
        use_cfg=False,
        look_ahead_tokens=12,
    )

    return detokenizer


def detokenize(detokenizer, tokens, ref_wav, ref_tokens):
    with torch.no_grad():
        detokenizer.clear_states()
        detokenizer.prefill(ref_wav, ref_tokens, chunk_size=150)
        cache_speech_collection = []
        chunk_size = 150
        first_chunk_size = 100
        first_chunk_tokens = tokens[:, :first_chunk_size]
        gen_speech = detokenizer.detokenize_streaming(
            first_chunk_tokens, is_final=tokens.size(1) <= first_chunk_size
        )
        cache_speech_collection.append(gen_speech)
        res_tokens = tokens[:, first_chunk_size:]
        for i in range(0, res_tokens.size(1), chunk_size):
            chunk_tokens = res_tokens[:, i : i + chunk_size]
            gen_speech = detokenizer.detokenize_streaming(
                chunk_tokens, is_final=(i + chunk_size >= res_tokens.size(1))
            )
            cache_speech_collection.append(gen_speech)

        gen_speech_all = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech_all


def detokenize_streaming(detokenizer, tokens, ref_wav, ref_tokens):
    with torch.no_grad():
        detokenizer.clear_states()
        detokenizer.prefill(ref_wav, ref_tokens, chunk_size=150)
        cache_speech_collection = []
        chunk_size = 150
        first_chunk_size = 100
        first_chunk_tokens = tokens[:, :first_chunk_size]
        gen_speech = detokenizer.detokenize_streaming(
            first_chunk_tokens, is_final=tokens.size(1) <= first_chunk_size
        )
        yield gen_speech
        res_tokens = tokens[:, first_chunk_size:]
        for i in range(0, res_tokens.size(1), chunk_size):
            chunk_tokens = res_tokens[:, i : i + chunk_size]
            gen_speech = detokenizer.detokenize_streaming(
                chunk_tokens, is_final=(i + chunk_size >= res_tokens.size(1))
            )
            yield gen_speech


def detokenize_noref(detokenizer, tokens):
    with torch.no_grad():
        detokenizer.clear_states()
        cache_speech_collection = []
        chunk_size = 150
        first_chunk_size = 100
        first_chunk_tokens = tokens[:, :first_chunk_size]
        gen_speech = detokenizer.detokenize_streaming(
            first_chunk_tokens, is_final=tokens.size(1) <= first_chunk_size
        )
        cache_speech_collection.append(gen_speech)
        res_tokens = tokens[:, first_chunk_size:]
        for i in range(0, res_tokens.size(1), chunk_size):
            chunk_tokens = res_tokens[:, i : i + chunk_size]
            gen_speech = detokenizer.detokenize_streaming(
                chunk_tokens, is_final=(i + chunk_size >= res_tokens.size(1))
            )
            cache_speech_collection.append(gen_speech)

        gen_speech_all = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech_all


def detokenize_noref_streaming(detokenizer, tokens):
    with torch.no_grad():
        detokenizer.clear_states()
        cache_speech_collection = []
        chunk_size = 150
        first_chunk_size = 100
        first_chunk_tokens = tokens[:, :first_chunk_size]
        gen_speech = detokenizer.detokenize_streaming(
            first_chunk_tokens, is_final=tokens.size(1) <= first_chunk_size
        )
        yield gen_speech
        res_tokens = tokens[:, first_chunk_size:]
        for i in range(0, res_tokens.size(1), chunk_size):
            chunk_tokens = res_tokens[:, i : i + chunk_size]
            gen_speech = detokenizer.detokenize_streaming(
                chunk_tokens, is_final=(i + chunk_size >= res_tokens.size(1))
            )
            yield gen_speech
