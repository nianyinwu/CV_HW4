""" PromptIR model """
## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090
## https://github.com/va1shn9v/PromptIR

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

##########################################################################
## Layer Norm

def to_3d(x):
    """
    Reshapes 4D tensor (B, C, H, W) into 3D tensor (B, H*W, C) for layer normalization.

    Args:
        x (Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        Tensor: Reshaped tensor of shape (B, H*W, C).
    """

    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    """
    Reshapes 3D tensor (B, H*W, C) back to 4D tensor (B, C, H, W).

    Args:
        x (Tensor): Input tensor of shape (B, H*W, C).
        h (int): Height.
        w (int): Width.

    Returns:
        Tensor: Reshaped tensor of shape (B, C, H, W).
    """

    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    """
    Layer normalization without bias, applied over the last dimension.
    """

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        Forward Pass
        """
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """
    Standard layer normalization with learnable bias and scale parameters.
    """

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        Forward Pass
        """
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """
    Applies BiasFree or WithBias layer normalization to a 4D input tensor.

    Args:
        dim (int): Channel dimension.
        LayerNorm_type (str): Type of LayerNorm ('BiasFree' or 'WithBias').
    """

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """
        Forward Pass
        """
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    """
    Gated Depthwise Convolutional Feed-Forward Network (GDFN).

    Args:
        dim (int): Input and output dimension.
        ffn_expansion_factor (float): Expansion ratio for hidden channels.
        bias (bool): Whether to use bias in convolutions.
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features*2,
            hidden_features*2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features*2,
            bias=bias
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Forward Pass
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    """
    Multi-DConv Head Transposed Self-Attention (MDTA).

    Args:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        bias (bool): Whether to use bias in convolutions.
    """

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3,
            dim*3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim*3,
            bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        """
        Forward Pass
        """
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions and a PReLU activation.
    """

    def __init__(self, dim):

        super(resblock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        """
        Forward Pass
        """
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules

class Downsample(nn.Module):
    """
    Downsamples the input by a factor of 2 using PixelUnshuffle.

    Args:
        n_feat (int): Number of input features.
    """

    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        """
        Forward Pass
        """
        return self.body(x)

class Upsample(nn.Module):
    """
    Upsamples the input by a factor of 2 using PixelShuffle.

    Args:
        n_feat (int): Number of input features.
    """

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        """
        Forward Pass
        """
        return self.body(x)


##########################################################################
## Transformer Block

class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of attention and feed-forward modules.

    Args:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        ffn_expansion_factor (float): Expansion factor for FFN.
        bias (bool): Whether to use bias in convolutions.
        LayerNorm_type (str): Type of LayerNorm to use ('BiasFree' or 'WithBias').
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        """
        Forward Pass
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv

class OverlapPatchEmbed(nn.Module):
    """
    Embeds image patches using a 3x3 convolution with overlap.

    Args:
        in_c (int): Number of input channels.
        embed_dim (int): Number of output embedding dimensions.
        bias (bool): Whether to use bias in convolution.
    """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        """
        Forward Pass
        """
        x = self.proj(x)

        return x




##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    """
    Generates adaptive prompts conditioned on input features.

    Args:
        prompt_dim (int): Number of channels in each prompt component.
        prompt_len (int): Number of prompt components.
        prompt_size (int): Spatial size of each prompt component.
        lin_dim (int): Input dimension used for computing attention weights.
    """

    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)


    def forward(self,x):
        """
        Forward Pass
        """
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = (
            prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        )
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


##########################################################################
##---------- CBAM Module -----------------------

class CBAM(nn.Module):
    """
    Define CBAM module
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention Module
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward Pass
        """

        # Channel Attention Module
        avg_pool = self.mlp(self.avg_pool(x))
        max_pool = self.mlp(self.max_pool(x))
        channdel_out = self.channel_sigmoid(avg_pool+max_pool)
        x = x * channdel_out

        # Spatial Attention Module
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        spatial_out = self.spatial_sigmoid(spatial_out)
        out = x * spatial_out

        return out


##########################################################################
##---------- PromptIR -----------------------

class PromptIR(nn.Module):
    """
    PromptIR: Transformer-based image restoration network with prompt-guided decoding.

    Args:
        inp_channels (int): Number of input image channels.
        out_channels (int): Number of output image channels.
        dim (int): Base feature dimension.
        num_blocks (list[int]): Number of Transformer blocks at each decoder level.
        num_decoder_blocks (list[int]): Number of Transformer blocks at each encoder level.        
        num_refinement_blocks (int): Number of refinement blocks after decoding.
        heads (list[int]): Number of attention heads at each level.
        ffn_expansion_factor (float): Expansion factor for feed-forward networks.
        bias (bool): Whether to use bias in convolutions.
        LayerNorm_type (str): Type of LayerNorm ('WithBias' or 'BiasFree').
        decoder (bool): Whether to use prompt-guided decoder or not.
    """
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [4,6,6,8],
        num_decoder_blocks=[8, 8, 8],
        num_refinement_blocks = 8,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = True,
    ):

        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)


        self.decoder = decoder

        if self.decoder:
            self.prompt1 = PromptGenBlock(
                prompt_dim=64,
                prompt_len=5,
                prompt_size = 64,
                lin_dim = 96
            )
            self.prompt2 = PromptGenBlock(
                prompt_dim=128,
                prompt_len=5,
                prompt_size = 32,
                lin_dim = 192
            )
            self.prompt3 = PromptGenBlock(
                prompt_dim=320,
                prompt_len=5,
                prompt_size = 16,
                lin_dim = 384
            )


        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)

        self.cbam1 = CBAM(dim)
        self.cbam2 = CBAM(dim * 2)
        self.cbam3 = CBAM(dim * 4)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(
            int(dim*2**1) + 128,
            int(dim*2**1),
            kernel_size=1,
            bias=bias
        )
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(
            int(dim*2**2) + 256,
            int(dim*2**2),
            kernel_size=1,
            bias=bias
        )
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim*2**1)+192,
            int(dim*2**2),
            kernel_size=1,
            bias=bias
        )
        self.noise_level3 = TransformerBlock(
            dim=int(dim*2**2) + 512,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )
        self.reduce_noise_level3 = nn.Conv2d(
            int(dim*2**2)+512,
            int(dim*2**2),
            kernel_size=1,
            bias=bias
        )


        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_decoder_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2**2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2),
            int(dim * 2**1),
            kernel_size=1,
            bias=bias
        )

        self.noise_level2 = TransformerBlock(
            dim=int(dim * 2**1) + 224,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        self.reduce_noise_level2 = nn.Conv2d(
            int(dim * 2**1) + 224,
            int(dim * 2**2),
            kernel_size=1,
            bias=bias
        )

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_decoder_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1

        self.noise_level1 = TransformerBlock(
            dim=int(dim * 2**1) + 64,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        self.reduce_noise_level1 = nn.Conv2d(
            int(dim * 2**1) + 64,
            int(dim * 2**1),
            kernel_size=1,
            bias=bias
        )

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_decoder_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(
            int(dim * 2**1),
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )


    def forward(self, inp_img):
        """
        Forward Pass
        """

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # Apply CBAM1
        out_enc_level1 = self.cbam1(out_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # Apply CBAM2
        out_enc_level2 = self.cbam2(out_enc_level2)


        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # Apply CBAM3
        out_enc_level3 = self.cbam3(out_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            dec3_param = self.prompt3(latent)

            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:

            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
