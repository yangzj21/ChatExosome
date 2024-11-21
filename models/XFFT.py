import logging
import math
from typing import Tuple, Optional, List, Union, Any, Type, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.layers import DropPath
# from timm.layers import Mlp
from einops import rearrange
from functools import partial
from itertools import repeat
import collections.abc

__all__ = ['FFT']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def get_linear_layer(use_conv, **kwargs):
    if use_conv:
        groups = kwargs.get('groups', 1)
        linear_layer = partial(nn.Conv1d, kernel_size=1, groups=groups)
    else:
        linear_layer = nn.Linear
    return linear_layer


class Mlp(nn.Module):
    r"""
    Alex: supported groups when use_conv=True
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer=None,
            bias=True,
            drop: Union[float, Tuple[float, ...]] = 0.,
            use_conv=False,
            **kwargs
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = get_linear_layer(use_conv, **kwargs)

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiHeadAttention(nn.Module):
    r"""
    Args:
        dim (int): Number of input features
        num_heads (int): Number of attention heads
        window_size (int): Window size
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int,
            drop_attn: float = 0.0,
            drop_proj: float = 0.0,
            meta_hidden_dim: int = 384,
            sequential_attn: bool = False,
            **kwargs
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (num_heads)."
        self.in_features: int = dim
        self.num_heads: int = num_heads
        self.window_size: int = window_size
        self.sequential_attn: bool = sequential_attn

        groups = kwargs.get('groups', 1)
        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)

        # NOTE old checkpoints used inverse of logit_scale ('tau') following the paper, see conversion fn
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))

        self.positional_encode = kwargs.get('positional_encode', True)
        # meta network for positional encodings
        if self.positional_encode:
            self.meta_mlp = Mlp(
                2,  # x, y
                hidden_features=meta_hidden_dim,
                out_features=num_heads,
                act_layer=nn.ReLU,
                drop=(0.125, 0.)  # FIXME should there be stochasticity, appears to 'overfit' without?
            )
            self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) -> None:
        """Method initializes the pair-wise relative positions to compute the positional biases."""
        device = self.logit_scale.device
        coordinates = torch.stack(torch.meshgrid([
            torch.arange(1, device=device),
            torch.arange(self.window_size, device=device)], indexing='ij'), dim=0).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(
            1.0 + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log, persistent=False)

    def _relative_positional_encodings(self) -> torch.Tensor:
        """Method computes the relative positional encodings

        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        window_area = self.window_size * 1
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(
            self.num_heads, window_area, window_area
        )
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * nW, Mh*Mw, C)
                B: batch, nW: num_windows, w: window_size * 1
            mask (Optional[torch.Tensor]): Attention mask for the case.
                Alex: No use here. used in _shift_window_attention.

        Returns:
            Output tensor of the shape [B * nW, w, C]
        """
        B, L, C = x.shape

        # qkv.shape -> [3, B, nH, L, head_dim]
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        # compute attention map with scaled cosine attention
        attn = (F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1. / 0.01)).exp()

        # attn.shape -> [Bw, num_heads, Mh*Mw, Mh*Mw]
        attn = attn * logit_scale
        if self.positional_encode:
            attn = attn + self._relative_positional_encodings()

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(B, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFTBlock(nn.Module):
    r"""This class implements the block.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads to be utilized
        input_resolution (int): Input resolution
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        extra_norm (bool): Insert extra norm on 'main' branch if True
        sequential_attn (bool): If true sequential self-attention is performed
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            input_resolution: int,
            mlp_ratio: float = 4.0,
            init_values: Optional[float] = 0,
            proj_drop: float = 0.0,
            drop_attn: float = 0.0,
            drop_path: float = 0.0,
            extra_norm: bool = False,
            sequential_attn: bool = False,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            **kwargs
    ) -> None:
        super(FFTBlock, self).__init__()
        self.dim: int = dim
        self.input_resolution: int = input_resolution
        self.init_values: Optional[float] = init_values

        # attn branch
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=input_resolution,
            drop_attn=drop_attn,
            drop_proj=proj_drop,
            sequential_attn=sequential_attn,
            meta_hidden_dim=input_resolution * 4,
            **kwargs
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()

        # mlp branch
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
            out_features=dim,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()

        # Extra main branch norm layer mentioned for Huge/Giant models in V2 paper.
        # Also being used as final network norm and optional stage ending norm while still in a C-last format.
        self.norm3 = norm_layer(dim) if extra_norm else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # extra, module specific weight init
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, HW]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C, HW]
        """
        # post-norm branches: (op -> norm -> drop)

        # Alex: FFTBlock is different from SwinTransformerBlock, because there is no _shift_window_attention.
        #       It utilizes full input_resolution window attention.
        x = x + self.drop_path1(self.norm1(self.attn(x)))

        # x.shape -> B, C, HW
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = self.norm3(x)  # main-branch norm enabled for some blocks / stages (every 6 for Huge/Giant)
        return x


class PatchMerging1D(nn.Module):
    """ This class implements the patch merging as a strided convolution with a normalization before.
    Args:
        dim (int): Number of input channels
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized.
    """

    def __init__(self, dim: int, norm_layer: Type[nn.Module] = nn.LayerNorm) -> None:
        super(PatchMerging1D, self).__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(in_features=4 * dim, out_features=2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, L]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, L // 4]
        """
        B, L, C = x.shape
        assert L % 4 == 0, ''

        x = x.reshape(B, L // 4, 4, C).permute(0, 1, 3, 2).flatten(2)   # permute是为了使在channel维上分组时保持原来组序号
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed1D(nn.Module):
    """ 1D Spectrum to Patch Embedding """

    def __init__(self, spectrum_size=1024, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()
        self.spectrum_size = spectrum_size
        self.patch_size = patch_size
        self.grid_size = spectrum_size // patch_size
        self.num_patches = patch_size

        self.groups = kwargs.get('groups', 1)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, groups=self.groups)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        if L % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - L % self.patch_size))
        # _assert(L == self.spectrum_size, f"Input spectrum size ({L}) doesn't match model ({self.spectrum_size}).")
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # A
        return x


class FFTStage(nn.Module):
    r"""This class implements a stage of the Swin transformer including multiple layers.

    Args:
        embed_dim (int): Number of input channels
        depth (int): Depth of the stage (number of layers)
        downscale (bool): If true input is downsampled
        input_resolution (Tuple[int, int]): input feature map size (H, W)
        num_heads (int): Number of attention heads to be utilized
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
            self,
            embed_dim: int,
            depth: int,
            downscale: bool,
            num_heads: int,
            input_resolution: int,
            mlp_ratio: float = 4.0,
            init_values: Optional[float] = 0.0,
            proj_drop: float = 0.0,
            drop_attn: float = 0.0,
            drop_path: Union[List[float], float] = 0.0,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            extra_norm_period: int = 0,
            extra_norm_stage: bool = False,
            sequential_attn: bool = False,
            **kwargs
    ) -> None:
        super(FFTStage, self).__init__()
        self.downscale: bool = downscale
        self.grad_checkpointing: bool = False
        self.input_resolution = input_resolution
        output_resolution = input_resolution
        if downscale:
            output_resolution = input_resolution // 4

        if downscale:
            self.downsample = PatchMerging1D(embed_dim, norm_layer=norm_layer)
            embed_dim = embed_dim * 2
        else:
            self.downsample = nn.Identity()

        def _extra_norm(index):
            i = index + 1
            if extra_norm_period and i % extra_norm_period == 0:
                return True
            return i == depth if extra_norm_stage else False

        self.blocks = nn.Sequential(*[
            FFTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                input_resolution=output_resolution,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop,
                drop_attn=drop_attn,
                drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                extra_norm=_extra_norm(index),
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
                **kwargs
            )
            for index in range(depth)]
                                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, HW]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, HW // 4]
        """
        x = rearrange(x, 'b c l -> b l c')

        x = self.downsample(x)
        for block in self.blocks:
            # Perform checkpointing if utilized
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        x = rearrange(x, 'b l c -> b c l')
        return x


class FFT(nn.Module):
    r"""
    Args:
        spectrum_size (int): Input spectrum size (spectral wavelength).
        patch_size (int): Patch size.
        in_chans: Number of input channels.
        depths: Depth of the stage (number of layers).
        num_heads: Number of attention heads to be utilized.
        embed_dim: Patch embedding dimension.
        num_classes: Number of output classes.
        mlp_ratio:  Ratio of the hidden dimension in the FFN to the input channels.
        drop_rate: Dropout rate.
        proj_drop_rate: Projection dropout rate.
        attn_drop_rate: Dropout rate of attention map.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Type of normalization layer to be utilized.
        extra_norm_period: Insert extra norm layer on main branch every N (period) blocks in stage
        extra_norm_stage: End each stage with an extra norm layer in main branch
        sequential_attn: If true sequential self-attention is performed.
        num_groups: Number of groups in Channel.
            This means group sequence should not be changed in the channel dimension from beginning to end, except the
            last classifier layer. If data OD belongs to channel OC with group number g, the generated data ND with
            channel NC from data OD should belongs to group number g.
    """

    def __init__(
            self,
            spectrum_size: int = 1024,
            patch_size: int = 4,
            in_chans: int = 1,
            num_classes: int = 1000,
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            mlp_ratio: float = 4.0,
            init_values: Optional[float] = 0.,
            drop_rate: float = 0.1,
            proj_drop_rate: float = 0.1,
            attn_drop_rate: float = 0.1,
            drop_path_rate: float = 0.1,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            extra_norm_period: int = 0,
            extra_norm_stage: bool = False,
            sequential_attn: bool = False,
            num_groups: int = 1,
            **kwargs: Any
    ) -> None:
        super(FFT, self).__init__()

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.spectrum_size = spectrum_size
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.feature_info = []

        kwargs['groups'] = num_groups
        self.patch_embed = PatchEmbed1D(
            spectrum_size=spectrum_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            **kwargs
        )
        patch_grid_size: int = self.patch_embed.grid_size

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        in_dim = embed_dim
        in_scale = 1
        input_resolution = patch_grid_size
        for stage_idx, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stages += [FFTStage(
                embed_dim=in_dim,
                depth=depth,
                downscale=stage_idx != 0,
                input_resolution=input_resolution,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                drop_attn=attn_drop_rate,
                drop_path=dpr[stage_idx],
                extra_norm_period=extra_norm_period,
                extra_norm_stage=extra_norm_stage or (stage_idx + 1) == len(depths),  # last stage ends w/ norm
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
                **kwargs
            )]
            if stage_idx != 0:
                in_dim *= 2
                in_scale *= 2
                input_resolution = input_resolution // 4
            self.feature_info += [dict(num_chs=in_dim, reduction=4 * in_scale, module=f'stages.{stage_idx}')]

        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.num_features, num_classes)
        )

        init_weights(self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape_size = len(x.shape)
        assert shape_size == 2 or shape_size == 3, \
            "Input data's shape is not 1D. The shape should be (B, L) or (B, C, L)."
        if shape_size == 2:
            x = rearrange(x, 'b, l -> b, c, l', c=1)

        # x.shape => 'b, c, l'
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def init_weights(module: nn.Module, name: str = ''):
    # FIXME WIP determining if there's a better weight init
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        elif 'head' in name:
            nn.init.zeros_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
