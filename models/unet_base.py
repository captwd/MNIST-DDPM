import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    r"""
    【做什么】
    把"时间步 t"这个数字编码成一个向量（位置嵌入），
    让网络知道"现在被加噪到第几步了"。

    做法是经典的正弦/余弦位置编码，和 Transformer 里的很像：
      对每个维度 i，用 sin/cos 造一个周期不同的值。
    【原理】
      让网络能区分 t=5 和 t=500 这样的不同时刻。

    :param time_steps: 时间步张量，形状 (B,)
    :param temb_dim:   要生成的嵌入维度（必须是偶数）
    :return:           形状 (B, temb_dim) 的嵌入向量
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)：让每个维度频率不同
    # 生成 0,1,2,... 的一半维度索引，再除以 (temb_dim//2)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # time_steps (B,) -> 先扩成 (B, temb_dim//2)，再除以 factor
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    # 一半用 sin，一半用 cos，拼起来得到 (B, temb_dim)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    r"""
    【做什么】
    U-Net 的下采样块：一路"缩小尺寸、加深通道"，提取高阶特征。
    它由若干层组成，每层包含：
      1. ResNet 块（带时间嵌入的卷积）
      2. 自注意力块（让网络关注全局关系）
      3. 可选的下采样（2x2 平均池化，尺寸减半）

    :param in_channels:  输入通道数
    :param out_channels: 输出通道数
    :param t_emb_dim:    时间嵌入的维度
    :param down_sample:  是否在这一层做下采样（尺寸减半）
    :param num_heads:    注意力头数
    :param num_layers:   这个 block 里重复多少层
    """
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        # 第 1 次卷积块：GroupNorm + SiLU + Conv2d
        # 第一层输入 in_channels，后面的层都用 out_channels
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        # 把时间嵌入 t_emb 也变换到 out_channels，用于和特征相加
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        # 第 2 次卷积块：GroupNorm + SiLU + Conv2d
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        # 自注意力前的归一化
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
             for _ in range(num_layers)]
        )

        # 多头自注意力：让每个位置都能看到整张图的信息
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )
        # 残差连接用的 1x1 卷积：把输入变换到 out_channels 方便相加
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        # 下采样卷积（stride=2 让尺寸减半）；如果不下采样就用 Identity（什么都不做）
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):

            # -------- ResNet 块 --------
            resnet_input = out                       # 记住输入，留给残差连接
            out = self.resnet_conv_first[i](out)     # 第一次卷积
            # 把时间嵌入加到特征上（形状从 (B,d) 扩到 (B,d,1,1) 以便相加）
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)    # 第二次卷积
            out = out + self.residual_input_conv[i](resnet_input)   # 残差相加

            # -------- 自注意力块 --------
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)      # 展平成 (B,C,H*W)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)                        # (B,H*W,C) 给attention用
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)  # 自注意力
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn                                    # 残差相加

        out = self.down_sample_conv(out)   # 可选下采样
        return out


class MidBlock(nn.Module):
    r"""
    【做什么】
    U-Net 最底部的中间块：在分辨率最低、通道最多的地方做更多处理。
    结构是：
      1. ResNet 块
      2. 自注意力块（重复 num_layers 次）
      3. ResNet 块

    :param in_channels/out_channels/t_emb_dim：同上
    :param num_heads:   注意力头数
    :param num_layers:  中间注意力块重复的次数
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers+1)      # 注意：这里比 num_layers 多一个（前后各一个）
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers + 1)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers+1)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )

    def forward(self, x, t_emb):
        out = x

        # 第一个 ResNet 块
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # 中间：注意力块 + ResNet 块 交替
        for i in range(self.num_layers):

            # 自注意力块
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            # ResNet 块
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)

        return out


class UpBlock(nn.Module):
    r"""
    【做什么】
    U-Net 的上采样块：一路"放大尺寸、减少通道"，恢复分辨率并重建图像。
    每层包含：
      1. 上采样（转置卷积，尺寸×2）
      2. 和对应的下采样特征拼接（跳连）
      3. ResNet 块 + 自注意力块

    :param in_channels:  输入通道数（注意进入了上采样后是拼接，所以这里往往是双倍）
    :param out_channels: 输出通道数
    :param t_emb_dim:    时间嵌入维度
    :param up_sample:    是否上采样
    :param num_heads:    注意力头数
    :param num_layers:   每块重复层数
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        # 上采样卷积：转置卷积，stride=2 让尺寸翻倍；如果不上采样就用 Identity
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()

    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)            # 先放大尺寸
        x = torch.cat([x, out_down], dim=1)   # 和对应的下采样特征拼接（跳连）
        # 拼接后通道数翻倍，所以 in_channels 是双倍的

        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        return out


class Unet(nn.Module):
    r"""
    【做什么】
    整个去噪网络 ε_θ：输入带噪图 x_t 和时间 t，输出对噪声 ε 的预测。
    结构是经典 U-Net 的"下采样(编码) -> 底部 -> 上采样(解码)"，带跳连。

    所有超参都从 config 里的 model_params 读进来（见 config/default.yaml）。
    """
    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config['im_channels']          # 图片通道数（MNIST 为 1）
        self.down_channels = model_config['down_channels'] # 下采样各层通道数
        self.mid_channels = model_config['mid_channels']   # 底部中间层通道数
        self.t_emb_dim = model_config['time_emb_dim']      # 时间嵌入维度
        self.down_sample = model_config['down_sample']     # 各层是否下采样
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        # 一些尺寸匹配的检查，保证 U-Net 能"对上"
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        # 时间嵌入的投影：先做一个简单的 MLP 加工一下时间向量
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))  # 上采样和上采样要反过来
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

        # 下采样块序列
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                                        down_sample=self.down_sample[i], num_layers=self.num_down_layers))

        # 中间块序列
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                                      num_layers=self.num_mid_layers))

        # 上采样块序列（顺序是反过来的）
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, up_sample=self.down_sample[i], num_layers=self.num_up_layers))

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        r"""
        前向传播：
          - 输入 x：带噪图，形状 (B, C, H, W)
          - 输入 t：时间步，形状 (B,)
          - 输出：对噪声 ε 的预测，形状同 x

        流程：进入卷积 -> 生成时间嵌入 -> 下采样(保存每层特征) -> 底部处理 -> 上采样(用跳连拼接) -> 输出
        """
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W

        # 生成时间嵌入 t_emb -> B x t_emb_dim，再经过 MLP 加工
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []

        # -------- 下采样：记录每一层输出（留给上采样做跳连） --------
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4

        # -------- 底部中间块 --------
        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4

        # -------- 上采样：弹出对应下采样特征拼接（跳连） --------
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W  （预测的噪声）
        return out
