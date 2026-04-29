# Copied from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/almeval/models/kimi_audio.py

import sys
sys.path.insert(0, 'almeval/models/kimi_audio') #noqa
from kimia_infer.api.kimia import KimiAudio as KimiAudio_hf

from .base import BaseModel


class KimiAudio(BaseModel):
    NAME = 'Kimi-Audio'

    def __init__(self, model_path='moonshotai/Kimi-Audio-7B-Instruct', **kwargs):
        assert model_path is not None
        self.model = KimiAudio_hf(
            model_path=model_path, load_detokenizer=False)

        self.sampling_params = {
            'audio_temperature': 0.8,
            'audio_top_k': 10,
            'text_temperature': 0.0,
            'text_top_k': 5,
            'audio_repetition_penalty': 1.0,
            'audio_repetition_window_size': 64,
            'text_repetition_penalty': 1.1,
            'text_repetition_window_size': 16,
            'max_new_tokens': -1,  # TODO: set it
        }
        super().__init__()

    def get_prompt(self, msg: dict):
        return msg['text']

    def generate_inner(self, msg: dict):

        audio = msg['audio']

        if len(audio) == 1:
            audio = audio[0]
        else:
            raise NotImplementedError(
                f'Audio length {len(audio)} not supported')

        prompt = self.get_prompt(msg)

        messages = []

        if prompt is not None or prompt.strip() != '':
            messages.append(
                {'role': 'user', 'message_type': 'text', 'content': prompt})

        messages.append(
            {'role': 'user', 'message_type': 'audio', 'content': audio})

        _, text = self.model.generate(
            messages, **self.sampling_params, output_type='text')
        return prompt, text
