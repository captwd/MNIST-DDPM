import torch


class LinearNoiseScheduler:
    r"""
    【做什么】
    这是 DDPM 的"噪声调度器"，负责两件事：
      1. 正向过程：给一张干净图片 x0 按照时间步 t 加入噪声，得到 x_t
      2. 反向过程：根据模型预测的噪声，一步步"退"回干净图片
    它只负责数学运算，不包含神经网络，是扩散模型的核心数学部分。

    【对应论文】
    Ho et al. 2020 《DDPM》全文的 forward / reverse process 公式。
    """
    def __init__(self, num_timesteps, beta_start, beta_end):
        # 总共扩散多少步（论文里通常是 1000）
        self.num_timesteps = num_timesteps
        # 第 1 步的噪声强度（很小，几乎看不出来）
        self.beta_start = beta_start
        # 最后一步的噪声强度（很大，图片几乎变成纯噪声）
        self.beta_end = beta_end

        # betas: 每一步的噪声方差 beta_t，从 beta_start 线性增大到 beta_end
        # 对应论文里的 β_1 ... β_T
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # alphas: α_t = 1 - β_t  （保留图片信息的比例）
        self.alphas = 1. - self.betas

        # alpha_cum_prod: ᾱ_t = α_1 * α_2 * ... * α_t  （累积乘积）
        # 用它可以把"一步步加噪"折叠成"一步加噪"，是提速的关键
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

        # 提前算好开根号，后面用起来更快、更清晰
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)          # √ᾱ_t
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)  # √(1-ᾱ_t)

    def add_noise(self, original, noise, t):
        r"""
        正向加噪（训练时用）：
        直接打包算出一个"加到第 t 步"的带噪图片 x_t，不用真的一步步加。
        公式(论文式4)：
            x_t = √ᾱ_t * x0 + √(1-ᾱ_t) * ε
        其中 ε 就是我们要喂给网络去学的"真实噪声"。

        :param original: 干净图片 x0，形状 (B, C, H, W)
        :param noise:    随机高斯噪声 ε，形状同 original
        :param t:        当前时间步，形状 (B,)，每个样本各自在第几步
        :return:         带噪图片 x_t
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        # 取出这一批各自对应时间步 t 的系数 √ᾱ_t 和 √(1-ᾱ_t)
        # [t] 用 batch 里每个样本自己的 t 去索引
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # reshape 直到系数能从 (B,) 变成 (B,1,1,1)，好和 (B,C,H,W) 的图片逐元素相乘
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # 套公式：√ᾱ_t * x0 + √(1-ᾱ_t) * ε
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
        反向去噪（采样时用）：
        已知当前带噪图 x_t 和网络预测的噪声 noise_pred，
        反推"上一步"的 x_{t-1}，顺便算出对 x0 的估计。

        它返回两个东西：
          - x_{t-1}  : 下一步要用的带噪图（按论文式6/11采样）
          - x0_pred  : 对"干净原图"的估计（可以用来可视化看效果）

        :param xt:         当前时间步的带噪图，形状 (B,C,H,W)
        :param noise_pred: 网络预测出的噪声 ε_θ(x_t, t)
        :param t:          当前时间步（标量）
        :return:           (x_{t-1}, x0_pred)
        """
        # 先反解出对干净原图 x0 的估计：
        #   x0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)   # 限制在 [-1,1]，因为训练图片就是这个范围

        # 计算 x_{t-1} 的"均值"（期望）
        #   mean = (x_t - β_t/√(1-ᾱ_t) * ε) / √α_t
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        # 如果到了 t=0（最后一步），就没有更早的 x_{-1} 了，直接返回均值
        if t == 0:
            return mean, x0
        else:
            # 否则，x_{t-1} = 均值 + 一个高斯噪声，要保留一点"随机性"（论文式6）
            # 方差按论文式6计算
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)   # 随机噪声

            return mean + sigma * z, x0
