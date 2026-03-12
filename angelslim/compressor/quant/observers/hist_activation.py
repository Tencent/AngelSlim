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

from typing import Tuple

import numpy as np
import torch

from .base_observer import BaseObserver


class HistObserver(BaseObserver):
    def __init__(
        self,
        layer=None,
        quant_bits=8,
        bins_count=2048,
        percent=0.9999,
        sign=True,
        symmetric=True,
    ):
        super(HistObserver, self).__init__(quant_bits=quant_bits, sign=sign, symmetric=symmetric)
        self._layer = layer
        self._bins_count = bins_count
        self._percent = percent
        self._upsample_bin_count = 64

        self._hist_min = None
        self._hist_max = None
        self._hist = None

        self.step = 0
        self.dev = None
        self.dtype = None

    def forward(self, inputs):
        """Calculate forward pass."""
        if not self.dtype:
            self.dtype = inputs.dtype
            self.dev = inputs.device

        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = None

        inputs_np = inputs.to(torch.float32).cpu()
        if self._hist_min is None or self._hist_max is None:
            self._hist_min, self._hist_max = self._min_max(inputs_np)
            self._hist = self._init_hists(inputs_np)
        else:
            new_min, new_max, new_hist = self._update_min_max_and_hist(
                inputs_np,
                self._hist_min,
                self._hist_max,
                self._hist,
                self._bins_count,
                self._upsample_bin_count,
            )
            self._hist_min, self._hist_max = new_min, new_max
            self._hist = new_hist

        return inputs

    def cal_min_max(self):
        return self._cal_min_max_by_percent()

    def _cal_min_max_by_percent(self):
        """Calculate the min and max value of the tensor based on the histogram."""
        hist = self._hist / np.sum(self._hist, dtype=np.float64)
        cumsumed_hist = np.cumsum(hist)
        max_idx = np.argwhere(cumsumed_hist >= min(self._percent, cumsumed_hist[-1]))[0]
        min_idx = np.argwhere(cumsumed_hist >= max(1 - self._percent, cumsumed_hist[0]))[0]
        bin_width = (self._hist_max - self._hist_min) / hist.shape[0]
        _max = self._hist_min + float((max_idx - 0.5) * bin_width)
        _min = self._hist_min + float((min_idx - 0.5) * bin_width)

        _max = torch.tensor(_max, dtype=self.dtype, device=self.dev)
        _min = torch.tensor(_min, dtype=self.dtype, device=self.dev)

        return _min, _max

    def _init_hists(self, inputs):
        """Initialize the histogram instance based on a tensor."""
        _min, _max = self._min_max(inputs)
        hist = None
        if _max > _min:
            hist, _ = np.histogram(inputs.numpy(), bins=self._bins_count, range=(_min, _max))
            hist = hist.astype(np.float32)

        return hist

    def _min_max(self, inputs):
        """Get the min and max value of a tensor."""
        return float(torch.min(inputs).numpy()), float(torch.max(inputs).numpy())

    def _update_min_max_and_hist(
        self,
        tensor,
        origin_min,
        origin_max,
        origin_hist,
        bins_count,
        upsample_bins_count,
    ):
        """Update the histogram and its range based on
            the values of the target tensor.
        Args:
            tensor: The tensor used to update the histogram.
            origin_min(float): The minimum of the original histogram's range.
            origin_max(float): The max of the original histogram's range.
            origin_hist: The original histogram.
            bins_count(int): The number of histogram bins.
            upsample_bins_count(int):
                The number of upsampled bins used to extend the histogram.
        """
        _origin_min, _origin_max = origin_min, origin_max
        _new_min, _new_max = self._min_max(tensor)

        if (_new_max - _new_min) == 0.0:
            return _origin_min, _origin_max, origin_hist
        elif _origin_max - _origin_min == 0.0:
            new_hist, _ = np.histogram(tensor.numpy(), range=(_new_min, _new_max), bins=bins_count)
            new_hist = new_hist.astype(np.float32)
            return _new_min, _new_max, new_hist
        elif _new_max <= _origin_max and _new_min >= _origin_min:
            new_hist, _ = np.histogram(
                tensor.numpy(), range=(_origin_min, _origin_max), bins=bins_count
            )
            new_hist = new_hist.astype(np.float32)
            new_hist += origin_hist
            return _origin_min, _origin_max, new_hist
        else:
            _new_min = min(_new_min, _origin_min)
            _new_max = max(_new_max, _origin_max)
            (
                _new_min,
                _new_max,
                downsample_bins_count,
                start_bin_idx,
            ) = self._relax_min_max(
                _new_min,
                _new_max,
                _origin_min,
                _origin_max,
                bins_count,
                upsample_bins_count,
            )

            new_hist, _ = np.histogram(tensor.numpy(), range=(_new_min, _new_max), bins=bins_count)

            merged_histogram = self._merge_histograms(
                new_hist,
                origin_hist,
                upsample_bins_count,
                downsample_bins_count,
                start_bin_idx,
                bins_count,
            )
            return _new_min, _new_max, merged_histogram

    def _merge_histograms(
        self,
        new_hist: np.ndarray,
        origin_hist: np.ndarray,
        upsample_bins_count: int,
        downsample_bins_count: int,
        start_bin_idx: int,
        bins_count: int,
    ):
        """Merge two histograms."""
        upsampled_histogram = np.repeat(origin_hist, upsample_bins_count)
        expanded_hist = np.zeros((bins_count * downsample_bins_count), dtype=np.float32)
        expanded_hist[start_bin_idx : bins_count * upsample_bins_count + start_bin_idx] = (
            upsampled_histogram
        )

        cumsumed_hist = np.cumsum(expanded_hist, dtype=np.float64)[
            downsample_bins_count - 1 :: downsample_bins_count
        ]
        shift_cumsumed_hist = np.zeros((bins_count), dtype=np.float64)
        shift_cumsumed_hist[1:] = cumsumed_hist[0:-1]
        sampled_hist = (cumsumed_hist - shift_cumsumed_hist) / upsample_bins_count
        new_hist = new_hist.astype(np.float32)
        new_hist += sampled_hist.astype(np.float32)
        return new_hist

    def _relax_min_max(
        self, new_min, new_max, origin_min, origin_max, bins_count, upsample_bins_count
    ) -> Tuple[float, float, int, int]:
        """Relax the min and max values of the histogram."""
        _bin_width = (origin_max - origin_min) / (bins_count * upsample_bins_count)
        downsample_bins_count = int(np.ceil((new_max - new_min) / (bins_count * _bin_width)))
        error = downsample_bins_count * bins_count * _bin_width - (new_max - new_min)
        new_max += error
        start_bin_idx = round((origin_min - new_min) / _bin_width)
        return new_min, new_max, downsample_bins_count, start_bin_idx

    def cal_thresholds(self):
        """Compute thresholds for MAX function."""
        assert self._hist is not None
        self._min, self._max = self.cal_min_max()
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def quant_axis(self):
        """Return quantization axis."""
        return -1

    def scales(self):
        """Return output scales."""
        if self._scale is None:
            self.cal_thresholds()
        if self.dtype:
            self._scale = self._scale.type(self.dtype)
        return self._scale

    def zero_points(self):
        """Return output zero points."""
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point
