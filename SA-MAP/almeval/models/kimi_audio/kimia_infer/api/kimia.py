import os
import torch
from loguru import logger

from kimia_infer.models.detokenizer import get_audio_detokenizer
from .prompt_manager import KimiAPromptManager
from kimia_infer.utils.sampler import KimiASampler
from huggingface_hub import snapshot_download
from .modeling_moonshot_kimia import MoonshotKimiaForCausalLM

prune_method = os.environ.get('method')
remain_token_ratio = float(os.environ.get('remain_token_ratio', 1.0))

class KimiAudio(object):
    def __init__(self, model_path: str, load_detokenizer: bool = True):
        logger.info(f"Loading kimi-audio main model")

        if os.path.exists(model_path):
            # local path
            cache_path = model_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_path)
    
        logger.info(f"Looking for resources in {cache_path}")
        logger.info(f"Loading whisper model")
        self.alm = MoonshotKimiaForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.alm = self.alm.to(torch.cuda.current_device())

        model_config = self.alm.config
        self.kimia_token_offset = model_config.kimia_token_offset

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path, kimia_token_offset=self.kimia_token_offset
        )

        if load_detokenizer:
            logger.info(f"Loading detokenizer")
            # need to compile extension moudules for the first time, it may take several minutes.
            self.detokenizer = get_audio_detokenizer(cache_path)
        else:
            # in this case, you're not allowed to generate audio(wav)
            self.detokenizer = None

        self.extra_tokens = self.prompt_manager.extra_tokens
        self.kimia_text_audiodelaytokens = 6
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]

    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,  # input audio tokens
        text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        attentions: torch.Tensor = None,
        attn_key: torch.Tensor = None,
        output_type: str = "text",
    ):

        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        text_stream_is_finished = False
        previous_audio_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        text_previous_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()

        if prune_method == "base":
            decoder_position_ids = (
                torch.arange(
                    0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
                )
                .unsqueeze(0)
                .long()
            )
            last_position_id = decoder_input_audio_ids.shape[1] - 1
        elif prune_method in ["visionzip"]:
            contextual_num_ratio = 0.05
            dominant_num_ratio = remain_token_ratio - contextual_num_ratio
            expanded_audio_token = (int(is_continuous_mask.sum() * dominant_num_ratio) + max(int(is_continuous_mask.sum() * contextual_num_ratio), 1))
            seq_len = (expanded_audio_token + decoder_input_audio_ids.shape[1] - is_continuous_mask.sum().item())
            decoder_position_ids = (
                torch.arange(
                    0, seq_len, device=torch.cuda.current_device()
                )
                .unsqueeze(0)
                .long()
            )
            last_position_id = seq_len - 1
        elif prune_method in ["vispruner",
                              "fastadasp",
                              "fastadasp-attention",
                              "fastadasp-attention-dpp",
                              "atome-merge",
                              "atome-merge-attention",
                              "atome-merge-attention-dpp",
                              "SA-MAP",
                              ]:
            seq_len = (max(int(remain_token_ratio * is_continuous_mask.sum()), 2) + decoder_input_audio_ids.shape[1] - is_continuous_mask.sum().item()) 
            decoder_position_ids = (
                torch.arange(
                    0, seq_len, device=torch.cuda.current_device()
                )
                .unsqueeze(0)
                .long()
            )
            print(f'> preprocess num : {max(int(remain_token_ratio * is_continuous_mask.sum()), 2)}')
            last_position_id = seq_len - 1
        else:
            seq_len = (int(remain_token_ratio * is_continuous_mask.sum()) + decoder_input_audio_ids.shape[1] - is_continuous_mask.sum().item())
            decoder_position_ids = (
                torch.arange(
                    0, seq_len, device=torch.cuda.current_device()
                )
                .unsqueeze(0)
                .long()
            )
            last_position_id = seq_len - 1


        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        valid_text_length = 0
        valid_audio_length = 0

        for i in range(max_new_tokens):
            audio_logits, text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                attentions=attentions,
                attn_key=attn_key,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            # Sample text token using the sampler
            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )

            # Sample audio token using the sampler
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )

            if text_stream_is_finished:
                next_token_text.fill_(self.extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
            else:
                valid_text_length += 1

            text_previous_tokens[i : i + 1] = next_token_text

            if i < self.kimia_text_audiodelaytokens:
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1

            previous_audio_tokens[i : i + 1] = next_audio_token

            audio_stream_is_finished = next_audio_token.item() in self.eod_ids

            if (
                output_type == "text"
                and text_stream_is_finished
                or output_type == "both"
                and audio_stream_is_finished
            ):
                return_text_tokens = (
                    text_previous_tokens[:valid_text_length]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return_audio_tokens = (
                    previous_audio_tokens[
                        self.kimia_text_audiodelaytokens : valid_audio_length
                        + self.kimia_text_audiodelaytokens
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return return_audio_tokens, return_text_tokens
            else:
                decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                decoder_input_text_ids = next_token_text.unsqueeze(1)

                decoder_position_ids = (
                    torch.zeros(1, 1, device=torch.cuda.current_device())
                    .fill_(last_position_id + 1)
                    .long()
                    .view(1, 1)
                )
                last_position_id += 1

                decoder_input_whisper_feature = None
                decoder_is_continuous_mask = None

        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return_audio_tokens = (
            previous_audio_tokens[
                self.kimia_text_audiodelaytokens : valid_audio_length
                + self.kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        return return_audio_tokens, return_text_tokens

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
    ):

        assert output_type in ["text", "both"]

        history = self.prompt_manager.get_prompt(chats, output_type=output_type)

        audio_input_ids, text_input_ids, is_continuous_mask = history.to_tensor()
        audio_features = history.continuous_feature
        attentions = history.attentions
        attn_key = history.attn_key

        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                max_new_tokens = 7500 - audio_input_ids.shape[1]

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            attentions=attentions,
            attn_key=attn_key,
            output_type=output_type,
        )

        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]  #  filter out the illegal tokens

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset

        generated_text_tokens = [
            t for t in generated_text_tokens if t < self.kimia_token_offset
        ]
        generated_text = self.detokenize_text(generated_text_tokens)
        if self.detokenizer is not None and output_type == "both":
            generated_wav = self.detokenize_audio(generated_wav_tokens)
        else:
            generated_wav = None

        return generated_wav, generated_text

    def detokenize_audio(self, audio_tokens):
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states()
        chunk_size = 30  # hard-coded right now
        first_chunk_size = 30
        cache_speech_collection = []
        audio_tokens = audio_tokens.to(torch.cuda.current_device())
        audio_tokens = audio_tokens.long()
        num_audio_tokens = audio_tokens.size(1)
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)
