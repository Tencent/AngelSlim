# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import warnings
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from functools import partial

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

from angelslim.utils import rank0_print

from ......utils.utils import decide_device_for_distributed
from ......utils.lazy_imports import torchaudio, whisper, onnxruntime
from ....inference.models.eagle3.target.modeling_cosyvoice3_kv import mel_spectrogram
from ..chat_templates import ChatTemplateType
from ..data_utils import CosyVoice3DataCollatorWithPadding
from .base_dataset_builder import OnlineDatasetBuilder
from .dataset_builder_factory import DatasetBuilderFactory


@DatasetBuilderFactory.register("online", "TTS")
class OnlineTTSDatasetBuilder(OnlineDatasetBuilder):
    def __init__(
        self,
        tokenizer: Union[AutoTokenizer, AutoProcessor],
        max_length: int = 2048,
        shuffle_seed: int = 42,
        chat_template_type: ChatTemplateType = ChatTemplateType.QWEN3,
        display: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            tokenizer,
            max_length,
            shuffle_seed,
            chat_template_type,
            display,
        )
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.global_rank = int(os.getenv("RANK", -1))
        self.output_dir = kwargs["output_dir"]
        self.device = decide_device_for_distributed()

        self.model_path = kwargs["target_model_name_or_path"]
        if not os.path.exists(self.model_path):
            self.model_path = snapshot_download(self.model_path)

        if os.path.exists(os.path.join(self.model_path, "cosyvoice3.yaml")):
            self.model_name = "cosyvoice3"
            onnx_path = os.path.join(self.model_path, "speech_tokenizer_v3.onnx")
            self._init_audio_tokenizer_cosyvoice3(onnx_path)
            self.feat_extractor = partial(mel_spectrogram,
                                        n_fft=1920,
                                        num_mels=80,
                                        sampling_rate=24000,
                                        hop_size=480,
                                        win_size=1920,
                                        fmin=0,
                                        fmax=None,
                                        center=False)

    def _init_audio_tokenizer_cosyvoice3(self, onnx_path) -> None:
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        providers = ["CUDAExecutionProvider"]
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            onnx_path, sess_options=option, providers=providers
        )

    def get_data_collator(self) -> Any:
        if self.model_name == "cosyvoice3":
            return CosyVoice3DataCollatorWithPadding()

    def read_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        try:
            for file in file_path:
                with open(file, "r", encoding="utf-8") as f:
                    for line in tqdm(
                        f,
                        desc=f"read data file {os.path.basename(file)}",
                        disable=self.global_rank > 0,
                    ):
                        try:
                            item = json.loads(line.strip())
                            if isinstance(item, dict):
                                data.append(item)
                        except json.JSONDecodeError as e:
                            warnings.warn(
                                f"JSON extract error: {e}, line: {line[:100]}..."
                            )
                            continue
        except Exception as e:
            rank0_print(f"read data file {file_path} failed: {e}")
        return data

    def build_dataset(
        self,
        datapath: str,
        num_proc: int = 8,
        shuffle: bool = True,
        sample_num: Optional[int] = None,
    ) -> Dataset:
        try:
            if not isinstance(datapath, list):
                datapath = [datapath]
            data_name = "_"
            for path in datapath:
                data_name += os.path.basename(path)[:-6]
            os.makedirs(self.output_dir, exist_ok=True)
            cache_path = os.path.join(
                self.output_dir, f"processed{data_name}_merged_cache.jsonl"
            )

            if not os.path.exists(cache_path):
                raw_data = self.read_jsonl_file(datapath)
                chunk_size = len(raw_data) // self.world_size
                start_idx = self.global_rank * chunk_size
                end_idx = (
                    start_idx + chunk_size
                    if self.global_rank < self.world_size - 1
                    else len(raw_data)
                )
                rank_data = raw_data[start_idx:end_idx]
                processed_data = []
                count = 0
                for item in tqdm(
                    rank_data,
                    desc=f"Rank {self.global_rank} process data",
                    disable=self.global_rank > 0,
                ):
                    if (
                        sample_num is not None
                        and count == sample_num // self.world_size
                    ):
                        break
                    text = item.get("text", "")
                    audio_tokens = item.get("audio_tokens", None)
                    audio_path = item.get("audio_path", "")
                    instruct = item.get("instruct", "")
                    instruct_audio_path = item.get("instruct_audio_path", "")

                    if self.model_name == "cosyvoice3":
                        processed = self._process_single_item_cosyvoice3(
                            text, audio_tokens, audio_path, instruct, instruct_audio_path
                        )
                    else:
                        raise NotImplementedError("This model is not implemented")

                    processed_data.append(processed)
                    count += 1

                # save for each rank
                rank_file = os.path.join(
                    self.output_dir,
                    f"processed{data_name}_rank_{self.global_rank}.jsonl",
                )
                with open(rank_file, "w", encoding="utf-8") as f:
                    for item in processed_data:
                        f.write(json.dumps(item, ensure_ascii=True) + "\n")
                done_file = os.path.join(
                    self.output_dir,
                    f"processed{data_name}_rank_{self.global_rank}.done",
                )
                Path(done_file).touch()
                self._wait_for_all_ranks_done(
                    self.output_dir, data_name, self.world_size
                )

                # merge processed data on rank 0
                merge_done_file = os.path.join(self.output_dir, f"processed{data_name}_merged_cache.done")
                if self.global_rank == 0:
                    all_processed_data = []
                    for rank in range(self.world_size):
                        rank_data = []
                        rank_tmp_file = os.path.join(
                            self.output_dir, f"processed{data_name}_rank_{rank}.jsonl"
                        )
                        with open(rank_tmp_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    rank_data.append(json.loads(line.strip()))
                        all_processed_data.extend(rank_data)

                    with open(cache_path, "w", encoding="utf-8") as f:
                        for item in all_processed_data:
                            f.write(json.dumps(item, ensure_ascii=True) + "\n")

                    for rank in range(self.world_size):
                        rank_tmp_file = os.path.join(
                            self.output_dir, f"processed{data_name}_rank_{rank}.jsonl"
                        )
                        rank_done_file = os.path.join(
                            self.output_dir, f"processed{data_name}_rank_{rank}.done"
                        )
                        if os.path.exists(rank_tmp_file):
                            os.remove(rank_tmp_file)
                        if os.path.exists(rank_done_file):
                            os.remove(rank_done_file)

                    with open(merge_done_file, 'w') as f:
                        f.write("Merged done")
                    rank0_print("Rank 0: Created merge completion marker")
                else:
                    merge_done = False
                    while not merge_done:
                        if os.path.exists(merge_done_file):
                            merge_done = True
                            break

                # Load dataset
                processed_ds = load_dataset("json", data_files=cache_path)

                # Conditionally shuffle dataset
                if shuffle:
                    processed_ds = processed_ds["train"].shuffle(seed=self.shuffle_seed)
                else:
                    processed_ds = processed_ds["train"]

                # Filter out None results with multiprocessing support
                processed_ds = processed_ds.filter(
                    lambda batch: [ids is not None for ids in batch["speech_token"]],
                    batched=True,
                    num_proc=num_proc,
                    desc="Filtering empty speech_token",
                )

                processed_ds.set_format(type="torch")
                return processed_ds
            
            else:
                # Load dataset
                rank0_print(f"Loading cache data from {cache_path}")
                ds = load_dataset("json", data_files=cache_path)

                # Conditionally shuffle dataset
                if shuffle:
                    ds = ds["train"].shuffle(seed=self.shuffle_seed)
                else:
                    ds = ds["train"]

                # Filter out None results with multiprocessing support
                ds = ds.filter(
                    lambda batch: [ids is not None for ids in batch["speech_token"]],
                    batched=True,
                    num_proc=num_proc,
                    desc="Filtering empty speech_token",
                )

                ds.set_format(type="torch")
                return ds

        except Exception as e:
            raise RuntimeError(f"Dataset building failed for {datapath}") from e

    def _wait_for_all_ranks_done(self, output_dir, data_name, world_size):
        all_done = False
        while not all_done:
            done_count = 0
            for rank in range(world_size):
                done_file = os.path.join(
                    output_dir, f"processed{data_name}_rank_{rank}.done"
                )
                if os.path.exists(done_file):
                    done_count += 1

            if done_count == world_size:
                all_done = True
                break

    def _process_single_item_cosyvoice3(
        self, text: str, audio_tokens: Optional[list], audio_path: str, instruct: Dict[str, Any], instruct_audio_path: str
    ) -> Optional[Dict[str, Any]]:
        text_token = self.tokenizer.encode(text)
        instruct_token = self.tokenizer.encode(instruct)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(instruct_audio_path)
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(instruct_audio_path)
        
        resample_rate = 24000
        if resample_rate == 24000:
            token_len = min(int(prompt_speech_feat.shape[1] / 2), prompt_speech_token.shape[1])
            prompt_speech_feat, prompt_speech_feat_len[:] = prompt_speech_feat[:, :2 * token_len], 2 * token_len
            prompt_speech_token, prompt_speech_token_len[:] = prompt_speech_token[:, :token_len], token_len

        if audio_tokens is not None:
            return {
                "text": text_token,
                "text_len": len(text_token),
                "speech_token": audio_tokens,
                "speech_token_len": len(audio_tokens),
                "prompt_speech_token": prompt_speech_token.squeeze(0).tolist(),
                "prompt_speech_token_len": prompt_speech_token_len.item(),
                "prompt_text": instruct_token,
                "prompt_text_len": len(instruct_token),
            }
        
        speech_token, speech_token_len = self._extract_speech_token(audio_path)
        return {
            "text": text_token,
            "text_len": len(text_token),
            "speech_token": speech_token.squeeze(0).tolist(),
            "speech_token_len": speech_token_len.item(),
            "prompt_speech_token": prompt_speech_token.squeeze(0).tolist(),
            "prompt_speech_token_len": prompt_speech_token_len.item(),
            "prompt_text": instruct_token,
            "prompt_text_len": len(instruct_token),
            }
    
    def _extract_speech_token(self, wav):
        speech = self.load_wav(wav, 16000)
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len
    
    def _extract_speech_feat(self, wav):
        speech = self.load_wav(wav, 24000)
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len
    
    def load_wav(self, wav, target_sr, min_sr=16000):
        speech, sample_rate = torchaudio.load(wav, backend='soundfile')
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech
