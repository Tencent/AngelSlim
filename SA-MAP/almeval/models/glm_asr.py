import os
from pathlib import Path

import torch
import torchaudio
from transformers import AutoTokenizer

from almeval.models.glmasr.configuration_glmasr import GlmasrConfig
from almeval.models.glmasr.modeling_glmasr import GlmasrModel
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

from .base import BaseModel

prune_method = os.environ.get('method')
remain_token_ratio = float(os.environ.get('remain_token_ratio', 1))

WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}

class GLMASRNano(BaseModel):
    NAME = 'GLM-ASR-Nano'

    def __init__(self, model_path='zai-org/GLM-ASR-Nano-2512', device="cuda", **kwargs):
        self.model_path = model_path
        self.device = device
        self.config = GlmasrConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = GlmasrModel.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        super().__init__()
    
    @staticmethod
    def build_prompt(
        msg,
        tokenizer,
        feature_extractor: WhisperFeatureExtractor,
        merge_factor: int,
        chunk_seconds: int =30,
    ) -> dict:
        def get_audio_token_length(seconds, merge_factor=2):
            def get_T_after_cnn(L_in, dilation=1):
                for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
                    L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                    L_out = 1 + L_out // stride
                    L_in = L_out
                return L_out

            mel_len = int(seconds * 100)
            audio_len_after_cnn = get_T_after_cnn(mel_len)
            audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1

            # TODO: current whisper model can't process longer sequence, maybe cut chunk in the future
            audio_token_num = min(audio_token_num, 1500 // merge_factor)

            return audio_token_num
        
        audio_path = msg['audio']
        if len(audio_path) == 1:
            audio_path = audio_path[0]
        audio_path = Path(audio_path)
        wav, sr = torchaudio.load(str(audio_path))
        wav = wav[:1, :]
        if sr != feature_extractor.sampling_rate:
            wav = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)(wav)

        tokens = []
        tokens += tokenizer.encode("<|user|>")
        tokens += tokenizer.encode("\n")

        audios = []
        audio_offsets = []
        audio_length = []
        chunk_size = chunk_seconds * feature_extractor.sampling_rate
        # for start in range(0, wav.shape[1], chunk_size):
        start = 0
        chunk = wav[:, start : start + chunk_size]
        mel = feature_extractor(
            chunk.numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]
        audios.append(mel)
        seconds = chunk.shape[1] / feature_extractor.sampling_rate
        num_tokens_ori = get_audio_token_length(seconds, merge_factor)
        if prune_method == 'visionzip':
            contextual_num_ratio = 0.05
            dominant_num_ratio = remain_token_ratio - contextual_num_ratio
            num_tokens = max(int(num_tokens_ori * dominant_num_ratio),1) + max(int(num_tokens_ori * contextual_num_ratio), 1)
        else:
            num_tokens = int(remain_token_ratio * num_tokens_ori)
        tokens += tokenizer.encode("<|begin_of_audio|>")
        audio_offsets.append(len(tokens))
        tokens += [0] * num_tokens
        tokens += tokenizer.encode("<|end_of_audio|>")
        audio_length.append(num_tokens_ori)

        if not audios:
            raise ValueError("音频内容为空或加载失败。")

            
        meta = msg['meta']
        if meta['task'] == 'ASR':
            assert 'lang' in meta
            lang = meta['lang']
            # from jsonl in: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
            prompt = f'Detect the language and recognize the speech: <|{lang}|>'
        elif meta['dataset_name'] == 'meld':
            # from: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
            prompt = 'Recognize the emotion with keywords in English:'
        elif meta['dataset_name'] == 'vocalsound':
            # from: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
            prompt = 'Classify the human vocal sound to VocalSound in English:'
        # help to invoke baesmodel continuous output
        elif meta['interactive'] == 'Audio-QA':
            prompt = ' Your answer to the question is:'
        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'Listen to the given audio carefully and answer this question: {msg["text"]} Your answer is:'
        else:
            prompt = msg['text'] + ' The answer is:'

    
        tokens += tokenizer.encode("<|user|>")
        tokens += tokenizer.encode("\nPlease transcribe this audio into text")
        # tokens += tokenizer.encode(prompt)
        tokens += tokenizer.encode("<|assistant|>")
        tokens += tokenizer.encode("\n")

        batch = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "audios": torch.cat(audios, dim=0),
            "audio_offsets": [audio_offsets],
            "audio_length": [audio_length],
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        }
        return batch
    
    @staticmethod
    def prepare_inputs(batch: dict, device: torch.device) -> tuple[dict, int]:
        tokens = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audios = batch["audios"].to(device)
        model_inputs = {
            "inputs": tokens,
            "attention_mask": attention_mask,
            "audios": audios.to(torch.bfloat16),
            "audio_offsets": batch["audio_offsets"],
            "audio_length": batch["audio_length"],
        }
        return model_inputs, tokens.size(1)

    
    def get_prompt(self, msg: dict):
        return msg['text']

    def generate_inner(self, msg: dict):
        feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
        batch = self.build_prompt(
            msg,
            self.tokenizer,
            feature_extractor,
            merge_factor=self.config.merge_factor,
        )
        model_inputs, prompt_len = self.prepare_inputs(batch, self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        transcript_ids = generated[0, prompt_len:].cpu().tolist()
        text = self.tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
        return self.tokenizer.decode(batch['input_ids'][0]), text




    


    


    

