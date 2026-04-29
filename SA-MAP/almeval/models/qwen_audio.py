# Modified from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/almeval/models/qwen_audio.py

import random

import librosa
import torch

from ..utils.misc import print_once
from .base import BaseModel

import sys
sys.path.insert(0, 'almeval/models/qwen2_audio')
from processing_qwen2_audio_pruner import Qwen2AudioProcessor
from qwen2_audio_pruner import Qwen2AudioForConditionalGeneration


class Qwen2Audio(BaseModel):
    NAME = 'Qwen2-Audio-7B'

    def __init__(self, model_path='Qwen/Qwen2-Audio-7B', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.processor = Qwen2AudioProcessor.from_pretrained(
            model_path
        )

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, device_map='cuda', attn_implementation='eager').eval()
        random.seed(0)
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        meta = msg['meta']
        if meta['task'] == 'ASR':
            assert 'lang' in meta
            lang = meta['lang']
            # from jsonl in: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
            prompt = f'Detect the language and recognize the speech: <|{lang}|>'
        elif meta['dataset_name'] == 'MELD':
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

        return '<|audio_bos|><|AUDIO|><|audio_eos|>' + prompt

    # 该模型是评测主力，只有chat才用chat模型
    def generate_inner(self, msg: dict):
        audio = None
        # 从message中提取audio和text
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]
        prompt = self.get_prompt(msg)

        print_once(f'Prompt: {prompt}')
        audio = librosa.load(
            audio, sr=self.processor.feature_extractor.sampling_rate)[0]

        inputs = self.processor(
            text=prompt,
            audio=audio,
            return_tensors='pt',
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        inputs = inputs.to('cuda')
        generated_ids = self.model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False,
                                            top_k=None,
                                            top_p=None)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        pred = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return prompt, pred


class Qwen2AudioChat(BaseModel):
    NAME = 'Qwen2-Audio-7B-Instruct'

    def __init__(self, model_path='Qwen/Qwen2-Audio-7B-Instruct', **kwargs):
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, device_map='cuda'
        )
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        meta = msg['meta']
        if meta['audio_type'] == 'AudioEvent':
            prompt = f'Listen to the given audio carefully and answer this question: {msg["text"]}.'
        else:
            prompt = msg['text']
        return prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]

        prompt = ''
        if msg['meta']['interactive'] == 'Audio-QA':
            conversation = [{'role': 'user',
                             'content': [{'type': 'audio',
                                          'audio_url': audio}]}]
        else:
            prompt = self.get_prompt(msg)
            # from: https://github.com/QwenLM/Qwen2-Audio/blob/dfc7d31b0a3181c8be496155bbf9eb3049499b3c/README.md?plain=1#L134
            conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user',
                             'content': [{'type': 'audio', 'audio_url': audio},
                                         {'type': 'text', 'text': prompt}]}]
        #
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios = []
        for message in conversation:
            if isinstance(message['content'], list):
                for ele in message['content']:
                    if ele['type'] == 'audio':
                        audios.append(
                            librosa.load(
                                ele['audio_url'],
                                sr=self.processor.feature_extractor.sampling_rate,
                            )[0]
                        )
        inputs = self.processor(
            text=text,
            audios=audios,
            return_tensors='pt',
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        inputs = inputs.to('cuda')
        generate_ids = self.model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        answer = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return prompt, answer
