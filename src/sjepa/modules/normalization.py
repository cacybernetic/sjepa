"""Normalization layers that run in float32 for stable training.

Speech models often train in half precision (fp16 or bf16) for speed. But
normalization is sensitive to small numbers, so it is safer to do the math in
float32 and then cast the result back. These two classes do exactly that.

Each class has one and only one job: normalize a tensor in float32.
"""

import torch.nn as nn
import torch.nn.functional as F


class Fp32GroupNorm(nn.GroupNorm):
    """Group normalization computed in float32.

    The input may be fp16 or bf16. We cast to float32, normalize, then cast
    back to the original type. This keeps the statistics stable.
    """

    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        output = F.group_norm(x.float(), self.num_groups, weight, bias, self.eps)
        return output.to(x.dtype)


class Fp32LayerNorm(nn.LayerNorm):
    """Layer normalization computed in float32.

    Same idea as `Fp32GroupNorm`: do the math in float32 and cast back. This
    is the standard trick used by HuBERT and wav2vec 2.0.
    """

    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        output = F.layer_norm(x.float(), self.normalized_shape, weight, bias,
                              self.eps)
        return output.to(x.dtype)
