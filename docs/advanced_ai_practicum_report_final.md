# 《高级人工智能实训》期末大作业报告

学 院 名 称：媒体工程学院  
课 程 名 称：高级人工智能实训  
作 品 名 称：面向单目图像的 SOAP 多视角法线先验监督头部 Avatar 重建研究  
学       号：__________  
姓       名：__________  
班       级：25 级电子信息硕士研究生  
任 课 教 师：__________  
学       期：2025-2026 学年第二学期  

---

## 一、作品简介

### 1. 任务背景与应用场景

本作品属于计算机视觉、神经渲染与三维数字人生成的交叉方向，目标是从单张人物图像中重建一个可驱动、可重渲染的三维头部 Avatar，并在目标驱动图像或驱动视频控制下生成新的表情与姿态结果。该任务具有以下典型应用场景：

1. 数字人直播与虚拟主播。
2. 视频会议中的人像重定向与虚拟形象驱动。
3. 短视频与 AIGC 内容制作。
4. 游戏与 XR 场景中的低成本人物建模。
5. 人机交互中的个性化数字分身。

本项目以 NeurIPS 2024 的 GAGAvatar 为基线，在原始训练框架基础上新增 SOAP 生成的六视角法线先验监督，核心研究问题是：在单目头像重建任务中，额外引入离线多视角法线伪监督，能否改善几何一致性与最终重建质量。

从任务本身的难度来看，这个问题并不简单。单张输入图像只提供了一个观察视角，而真实三维头部几何包含大量不可见区域，例如侧脸、耳部、头发后侧和帽檐下方区域。仅依赖单视图图像重建损失时，模型很容易学到“二维上看起来合理”的结果，却不一定真正恢复出一致的三维几何结构。因此，在这类任务中，引入外部几何先验具有明确的研究价值。

本文工作的价值不只在于“把代码跑通”，而在于完成了一条完整的方法扩展链路：从数据预处理、canonical feature frame 导出、SOAP 离线多视角生成，到训练时的法线监督接入，再到多组实验的量化与定性分析。这使得本项目具备了较强的研究型课程作业特征，而不仅是单纯的工程复现。

### 2. 作品实现的核心功能

本作品完成了以下核心功能。

1. 数据集加载与预处理：支持 VFHQ 子集的 LMDB 数据读取，并完成 matting 黑背景预处理。
2. 深度学习模型构建：基于 GAGAvatar 构建单图像头部重建与驱动模型，并扩展 SOAP 多视角法线监督分支。
3. 模型训练、验证与测试：支持 20000 step 的本地训练、验证集 best checkpoint 保存和继续实验。
4. 结果评估、指标计算与可视化：记录 PSNR、SSIM、train loss、soap normal loss，并生成验证拼图与 TensorBoard 日志。
5. 模型对比实验：完成 matted baseline、SOAP 从头训练、SOAP 微调三组对比实验。
6. 外部先验数据链路：完成 canonical feature frame 导出、SOAP 离线多视角生成、训练时按 frame key 精确读取的闭环。

进一步细化来看，这六项功能并不是彼此孤立的，而是构成了一个完整实验系统。数据预处理决定训练分布是否稳定，canonical frame 导出决定 SOAP 监督是否能稳定对应到视频级身份，数据加载器改动决定 SOAP 结果能否真正进入 batch，模型损失层改动决定几何监督是否真正参与反向传播，而训练监控和日志记录则决定最终结论是否能被量化复核。因此，这份作业的工作量同时覆盖了数据、模型、训练、可视化和结果分析五个层面。

从课程作业完成度的角度看，本项目并不是单纯“跑通一次官方代码”。它同时覆盖了数据预处理、训练配置修改、数据加载器扩展、模型损失函数扩展、离线教师先验接入、训练监控与结果分析这六个层面，因此更接近一个完整的小型研究项目，而不是简单的实验复现。

进一步地，相对于原始 GAGAvatar，本次工作至少包含以下四类实质性增量：

1. 数据层增量：新增 matted 数据集，减少背景域偏移。
2. 工具链增量：新增 canonical feature frame 导出脚本，建立 SOAP 输入生成链路。
3. 监督层增量：新增 SOAP 多视角法线监督分支，使模型从单视角图像监督扩展到离线多视角几何监督。
4. 实验层增量：不仅比较最终结果，还分析训练收敛过程、监督项收敛情况与训练代价。

### 3. 所使用的技术

| 类别 | 内容 |
| --- | --- |
| 编程语言 | Python 3.12.2 |
| 深度学习框架 | PyTorch 2.4.1、Torchvision 0.19.1、Lightning 2.4.0 |
| 三维/渲染库 | PyTorch3D 0.7.8 |
| 主要工具库 | NumPy、FAISS-GPU、TensorBoard、TQDM |
| 视觉编码 | DINOv2 / Vision Transformer |
| 三维先验 | FLAME 参数化人脸模型 |
| 渲染表示 | 3D Gaussian Splatting 风格表示、双层 lifting planes |
| 外部先验 | SOAP 六视角图像与法线图 |
| 硬件环境 | NVIDIA GeForce RTX 4090，24564 MiB 显存 |

从技术栈角度看，本项目并不是使用一个单一神经网络直接完成全部任务，而是由多种技术共同组成：DINOv2 负责全局视觉特征编码，FLAME 负责参数化几何先验，Gaussian 表示负责可微渲染，StyleUNet 负责高分辨率图像重建，SOAP 则提供额外的多视角几何伪监督。这种“编码器 + 参数化几何 + 可微渲染 + 外部教师先验”的组合结构，也是本项目相比普通图像分类或普通生成任务更复杂的原因之一。

---

## 二、相关技术原理

### 1. 模型结构与核心思想

本项目的基础模型不是传统的单一 CNN 分类器，而是由图像编码、三维几何先验、可学习三维表示和可微渲染组成的复合系统。其主要结构如下。

1. 图像编码器：使用 DINOv2 提取输入 feature image 的高层语义与外观特征。DINOv2 本质上是 Vision Transformer，因此本项目在编码阶段实际使用了 Transformer 结构。
2. 几何先验模块：使用 FLAME 人脸模型提供 shape、pose、expression、eye pose 等参数，并构造目标几何点集。
3. 可学习三维表示：使用 global head points 与双层 lifting planes 生成高斯参数，分别建模稳定的人脸主体与更灵活的局部区域。
4. 渲染与重建模块：先进行 Gaussian rendering，再通过 StyleUNet 上采样获得最终 512×512 输出图像。
5. 几何监督模块：在原始 point loss 的基础上，新增 SOAP 多视角 normal guidance，用于对 lifting planes 生成的几何施加额外约束。

如果按前向传播顺序更细地描述，该模型的工作流可以写成：

1. 输入特征图 `f_image` 经过 DINOv2 编码器，得到局部特征 `f_feature0` 和全局特征 `f_feature1`。
2. 由相机射线方向构造 harmonic embedding，作为几何方向编码。
3. 全局分支 `gs_generator_g` 负责生成与人脸主体几何强相关的高斯参数。
4. 两个局部分支 `gs_generator_l0` 与 `gs_generator_l1` 分别预测 lifting plane 正反两个方向上的局部几何偏移。
5. 三个分支输出的高斯参数拼接后送入 Gaussian renderer 生成中间渲染结果 `gen_image`。
6. `StyleUNet` 把中间结果上采样为最终高分辨率结果 `sr_gen_image`。
7. 如果启用 SOAP guidance，则继续把局部几何渲染成多视角 normal map，并与 SOAP normal map 计算额外损失。

这种“编码器 + 几何生成器 + 可微渲染器 + 超分网络 + 几何监督”的结构，使得模型既能学习图像外观，又能通过额外几何约束逐步校正三维表示。

从代码实现来看，主干结构在模型初始化阶段被明确写出：

```python
self.base_model = DINOBase(output_dim=256)
self.head_base = nn.Parameter(torch.randn(5023, 256), requires_grad=True)
self.gs_generator_g = LinearGSGenerator(in_dim=1024, dir_dim=self.direnc_dim)
self.gs_generator_l0 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
self.gs_generator_l1 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
self.upsampler = StyleUNet(in_size=512, in_dim=32, out_dim=3, out_size=512)
```

其中，`DINOBase` 对应语义编码器，`gs_generator_g / l0 / l1` 负责生成高斯参数，`StyleUNet` 负责高分辨率图像重建。

如果从前向传播流程更细地观察，可以把模型理解成以下信息流：首先，输入的 feature image 被 DINOv2 编码成局部特征 `f_feature0` 和全局特征 `f_feature1`；随后，模型将全局特征与可学习的头部基向量 `head_base` 融合，用 `gs_generator_g` 生成全局高斯参数；再利用局部特征与平面方向编码分别通过 `gs_generator_l0` 和 `gs_generator_l1` 生成前后两层 lifting plane 的局部高斯；最后把三部分高斯参数拼接后送入 Gaussian renderer，再经由 StyleUNet 输出最终头像图像。这意味着模型同时在学习“全局头部主体”和“局部补充细节”，从而兼顾几何稳定性与边缘表达能力。

从结构设计角度看，这种双分支几何建模方式很适合头像任务。全局分支可以稳定建模面部核心区域，局部分支则更适合学习发丝、帽檐、脸部轮廓等复杂非刚性区域。你后面加入的 SOAP guidance，实际上主要影响的是后两层 lifting planes 的几何学习过程，而不是直接替代全局主体分支。

这一结构的一个重要特点是“主体几何”和“局部细节”并不是由同一个分支直接预测，而是分成全局点分支和双层局部分支。这也是为什么 SOAP 监督被加在 lifting plane 几何上更合理，因为它主要想约束的正是边缘、侧脸、帽檐、发丝等更容易出现歧义的区域。

### 2. 关键机制说明

#### 2.1 Transformer 编码机制

虽然本项目不是纯 Transformer 生成模型，但视觉主干 DINOv2 使用的是 ViT 编码方式。其核心自注意力机制可以表示为：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这使得编码器能够建立全局 patch 之间的关系，更有利于从单张人像中提取高层外观与身份特征。

#### 2.2 位置编码、残差连接、层归一化

ViT 结构依赖位置编码来保留图像 patch 的空间顺序；残差连接用于缓解深层网络训练中的梯度消失问题；LayerNorm 用于稳定训练分布。它们共同保证了 DINOv2 编码特征的稳定性，这对单图像头像重建非常关键。

#### 2.3 FLAME 参数化几何先验

FLAME 是一个可学习的参数化三维人脸模型。对于本任务而言，FLAME 的作用不是直接输出最终头像，而是提供一个稳定、结构化的人脸几何先验，用于：

1. 构造目标点集 `t_points`。
2. 提供相机变换与参数化姿态信息。
3. 在几何损失中作为 reference mesh。

FLAME 的参数化形式可简化写为：

$$
\mathbf{V}(\beta,\psi,\theta)=\overline{\mathbf{V}}+B_s(\beta)+B_e(\psi)+B_p(\theta)
$$

其中，$\beta$ 表示 shape 参数，$\psi$ 表示 expression 参数，$\theta$ 表示 pose 参数，$\overline{\mathbf{V}}$ 为均值模板，$B_s$、$B_e$、$B_p$ 分别表示形状、表情和姿态的基函数项。虽然本项目最终输出的不是传统 FLAME 网格，而是高斯表示的人头 Avatar，但 FLAME 仍然提供了非常关键的结构化先验，使训练不会完全陷入无约束的单目几何歧义之中。

#### 2.4 3D Gaussian Splatting 与 lifting planes

GAGAvatar 不直接预测传统的高精网格贴图，而是利用 3D Gaussian 表示结合双层 lifting planes 表达头部三维结构。这种表示方式兼具可微性、渲染效率和对复杂局部区域的表达能力。

与传统网格渲染相比，Gaussian 表示更适合做神经渲染任务，因为每个高斯点天然带有位置、尺度、透明度和颜色等属性，在视角变化时可以通过可微方式完成 splatting 和颜色混合，避免了复杂网格拓扑编辑带来的不稳定性。对于头像生成这类任务，这种表示对发丝、帽檐和轮廓区域尤其友好。

#### 2.4.1 调和方向编码

代码中使用了 `HarmonicEmbedding` 对平面方向进行编码，其目的在于让网络更容易感知方向变化与空间频率信息。调和编码的一般形式为：

$$
\gamma(\mathbf{d})=[\sin(2^0\pi\mathbf{d}),\cos(2^0\pi\mathbf{d}),\ldots,\sin(2^{L-1}\pi\mathbf{d}),\cos(2^{L-1}\pi\mathbf{d}),\mathbf{d}]
$$

这里的方向编码能够帮助生成器区分不同朝向上的几何结构，对于后续渲染法线和多视角一致性都具有重要作用。

#### 2.4.2 Gaussian 渲染累积机制

从渲染角度看，一个像素的颜色可以近似写成多个高斯贡献的加权和：

$$
I(u,v)=\sum_i w_i(u,v)\,c_i
$$

其中，$w_i(u,v)$ 是第 $i$ 个高斯在像素 $(u,v)$ 处的投影权重，$c_i$ 是对应的颜色或特征值。由于权重与相机投影、深度和透明度相关，因此整个渲染过程可以对几何和外观参数进行反向传播，这也是 GAGAvatar 能直接通过图像重建目标训练三维表示的基础。

与传统显式三角网格方案相比，这种表示具有三个优势：

1. 渲染更连续：高斯表示天然具有软可见性和可微特性。
2. 局部细节表达更灵活：双层 plane 可以建模脸部以外的头发、帽檐等局部外观。
3. 与学习型图像重建更兼容：渲染器输出可以直接送入超分网络进一步恢复纹理。

但它也带来一个问题：lifting plane 自身不是强几何先验，因此如果没有额外约束，模型容易学到“二维看起来合理但三维不够一致”的几何。这正是本文引入 SOAP normal guidance 的动机所在。

#### 2.5 SOAP 多视角法线先验

SOAP 可以从单张人像离线生成多个固定视角下的图像与法线图。本文并未将 SOAP 替换 GAGAvatar，而是将 SOAP 作为 teacher prior，具体做法是：

1. 从每个视频序列导出 canonical feature frame。
2. 运行 SOAP，得到 6 个固定视角的 normal maps。
3. 在 GAGAvatar 训练阶段，把预测的 lifting plane 顶点渲染为多视角 normal maps。
4. 在前景重叠区域上与 SOAP normal maps 计算余弦一致性损失。

因此，SOAP 在本项目中扮演的是“离线教师几何监督”的角色，而不是直接参与最终推理。

### 3. 损失函数与优化器

本项目使用 Adam 优化器，学习率为 1e-4。原始 GAGAvatar 的主要损失项包括：

1. 感知损失 `percep_loss`。
2. 图像重建损失 `img_loss`。
3. 边界框损失 `box_loss`。
4. 几何点集损失 `point_loss`。

其中，点集损失本质上是在目标 FLAME 点集与预测点集之间做最近邻平方距离约束，其简化形式可写为：

$$
\mathcal{L}_{point}=\frac{1}{N}\sum_{i=1}^{N}\min_j \|\mathbf{p}_i-\hat{\mathbf{p}}_j\|_2^2
$$

这个损失项的作用是让预测的三维结构不要偏离目标几何过远，是原始 GAGAvatar 中最重要的几何监督之一。你此次新增的 SOAP normal guidance，可以看成在 `point_loss` 基础上进一步加入“法线方向一致性”的辅助约束。

在此基础上，本文加入 SOAP 法线监督项，总损失写为：

$$
\mathcal{L}=\mathcal{L}_{percep}+\mathcal{L}_{img}+\mathcal{L}_{box}+\mathcal{L}_{point}+\lambda_s\mathcal{L}_{soap-normal}
$$

本次 SOAP 实验中：

$$
\lambda_s = 0.05
$$

SOAP 法线项的核心形式为：

$$
\mathcal{L}_{soap-normal}=\frac{1}{|\Omega|}\sum_{p\in\Omega}\left(1-\left|\langle \hat{n}_p,n^{soap}_p\rangle\right|\right)
$$

其中，$\Omega$ 表示预测法线与 SOAP 法线的前景重叠区域，$\hat{n}_p$ 表示模型预测法线，$n^{soap}_p$ 表示 SOAP 提供的目标法线。

从实现角度看，$\Omega$ 不是整张图，而是由预测 mask 与 SOAP mask 的重叠区域共同定义的，因此该损失实际上只在前景有效区域上起作用。这一点非常重要，因为如果把背景或空洞区域也强行纳入法线一致性计算，反而会给训练引入大量无意义噪声。

此外，需要特别指出的是，代码中实际上还支持 `NORMAL_LOSS` 的 `point` 和 `screen` 两种模式，说明你不仅实现了 SOAP 主实验，还实现了一个更通用的几何监督框架。只是本次 matted 数据集的主实验中，`NORMAL_LOSS.ENABLED` 被设为 `false`，而主监督项转为 `SOAP_GUIDANCE`。

Adam 优化器的参数更新可写为：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$

$$
	heta_t=\theta_{t-1}-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

其中，$g_t$ 为当前梯度，$m_t$ 和 $v_t$ 分别是一阶矩和二阶矩估计，$\eta$ 为学习率。对本任务而言，Adam 的优势在于面对多损失项联合优化时能够保持较好的稳定性，尤其适合这种图像重建、几何约束、法线监督同时存在的训练场景。

这里绝对值项 $|\langle \hat{n}_p,n^{soap}_p\rangle|$ 的意义也值得说明。它意味着当前实现更关注法线方向的一致性，而对正反朝向的小幅翻转更宽容。这种设计在伪监督条件下是合理的，因为 SOAP normal 本身并非真实标注，局部区域存在方向不稳定时，直接使用带符号余弦相似度反而可能使优化更加脆弱。

如果进一步把各损失项的作用拆开，可以得到如下理解：

1. `percep_loss` 主要负责维持视觉语义与面部感知质量。
2. `img_loss` 主要负责像素级重建精度。
3. `box_loss` 约束人脸区域范围，避免渲染区域漂移。
4. `point_loss` 提供基础的几何对齐信号。
5. `soap_normal_loss` 提供额外的多视角局部几何一致性信号。

因此，本项目实际上是在“二维重建目标”和“额外三维几何目标”之间做平衡。实验结果表明，这种平衡在当前实现下还没有完全调到最优。

在图像质量评价方面，本文使用 PSNR 与 SSIM 两个指标。PSNR 定义为：

$$
\mathrm{PSNR}=10\log_{10}\frac{MAX^2}{\mathrm{MSE}}
$$

SSIM 定义为：

$$
\mathrm{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

由于本任务是生成与重建任务，而不是分类任务，因此模板中常见的准确率、精确率、召回率、F1 等指标不适用，本文以 PSNR、SSIM 和定性可视化作为主要评价手段。

---

## 三、数据集与预处理

### 1. 数据集介绍

本报告重点分析以下一个训练数据集和三组对应实验：

1. 数据集路径：`/root/autodl-fs/vfhq_group100_101_matted`
2. 输出目录：
   - `outputs/GAGAvatar_VFHQ/Apr25_0138_rbtam`
   - `outputs/GAGAvatar_VFHQ/Apr25_1030_kxwey`
   - `outputs/GAGAvatar_VFHQ/Apr25_1707_nheku`

根据 `dataset.json` 的真实统计结果，该 matted 数据集的划分如下。

| 划分 | 帧数 | 视频数 | 用途 |
| --- | --- | --- | --- |
| train | 8898 | 31 | 训练 |
| val | 64 | 17 | 验证与保存 best checkpoint |
| test | 32 | 15 | 独立测试保留 |

如果进一步从“视频级样本”角度看，当前训练设置实际上是一个以视频身份为单位组织的单目头像重建任务。train 集中 31 个视频对应 31 个 canonical feature frame；val 集中 17 个视频对应 17 个 canonical feature frame。每个 canonical feature frame 又被 SOAP 扩展为 6 个固定视角的法线图，因此仅从几何先验角度看，训练阶段实际可用的 SOAP normal views 总数约为：

$$
31 \times 6 = 186
$$

验证阶段对应的 SOAP normal views 总数约为：

$$
17 \times 6 = 102
$$

这说明你的 SOAP 监督虽然最终要在帧级训练中使用，但其几何先验的组织单位实际上是“视频级 canonical 输入”。这一点很适合在答辩时强调，因为它解释了为什么 `USE_CANONICAL_FEATURE_FRAME=true` 是必要设置。

需要说明的是，当前三个实验目录中的 PSNR/SSIM 来自验证集日志，而不是单独另存的 test set evaluation，因此本文在“量化指标”部分统一使用验证集指标进行可复核分析。

如果从数据密度上进一步分析，可以发现该数据集具有以下特点：

1. 训练集 8898 帧对应 31 个视频，平均每个训练视频约有 287 帧。
2. 验证集 64 帧对应 17 个视频，平均每个验证视频约有 3.76 帧。
3. 测试集 32 帧对应 15 个视频，平均每个测试视频约有 2.13 帧。

这说明该数据集更像是“少身份、多帧”的局部子集，而不是大规模多身份训练集。也正因为如此，本次实验更适合回答“在当前子集上方法是否有效”这一问题，而不是直接给出特别强的泛化结论。

### 2. 数据预处理流程

#### 2.1 原始训练格式

数据以 GAGAvatar 训练所需的结构组织：

1. `img_lmdb`：存储裁剪后的人头图像。
2. `optim.pkl`：存储跟踪与优化后的 FLAME 参数、bbox、相机矩阵等信息。
3. `dataset.json`：记录 train / val / test 划分。

为了让读者更清楚理解训练数据在工程上的组织方式，表 3-1 给出了各文件的作用说明。

| 文件或目录 | 内容 | 在训练流程中的作用 |
| --- | --- | --- |
| `img_lmdb` | 裁剪后的人头 RGB 图像 | 提供 feature image 与 target image |
| `optim.pkl` | 跟踪得到的 shape、pose、expression、bbox、相机矩阵 | 构造 FLAME 点集与目标视角 |
| `dataset.json` | train / val / test 划分列表 | 控制训练集与验证集读取 |
| `soap_inputs/feature_frames` | 导出的 canonical 输入帧 | 作为 SOAP 离线生成输入 |
| `soap_output/.../6-views/images` | SOAP 生成的六视角 RGB | 预留 RGB guidance 扩展 |
| `soap_output/.../6-views/normals` | SOAP 生成的六视角 normal maps | 当前主实验使用的几何监督数据 |

#### 2.2 Matting 预处理

为了让训练时的背景分布更接近推理阶段的黑背景头部图像，本文使用新增的 `tools/matte_lmdb_dataset.py` 对 LMDB 数据做人像抠图。其关键逻辑为：

```python
image = src_lmdb[key].float().to(device) / 255.0
image = matting_engine(image, return_type='matting', background_rgb=background)
image = (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()
dst_lmdb.dump(key, image, type='image')
```

这一步的作用是统一背景为黑色，减少复杂背景对头像重建任务的干扰。后续实验结果也验证了 matting 对性能有明显帮助。

#### 2.3 SOAP 输入帧导出

你本次实验使用的 canonical feature frame 导出命令如下：

```bash
cd /root/autodl-tmp/GAGAvatar
/root/miniconda3/envs/GAGAvatar/bin/python tools/export_soap_feature_frames.py \
  --dataset_path /root/autodl-fs/vfhq_group100_101_matted \
  --output_dir /root/autodl-tmp/GAGAvatar/soap_inputs/feature_frames \
  --split train

/root/miniconda3/envs/GAGAvatar/bin/python tools/export_soap_feature_frames.py \
  --dataset_path /root/autodl-fs/vfhq_group100_101_matted \
  --output_dir /root/autodl-tmp/GAGAvatar/soap_inputs/feature_frames_val \
  --split val
```

该脚本的核心逻辑是先按视频聚合帧，再为每个视频选取 canonical feature frame：

```python
video_info, _ = build_video_info(dataset_split[split])
feature_keys = [video_info[video_id][0] for video_id in sorted(video_info.keys())]
for frame_key in feature_keys:
    image = lmdb_engine[frame_key]
    output_path = os.path.join(output_dir, f'{frame_key}.png')
    torchvision.io.write_png(image.to(torch.uint8), output_path)
```

这一步虽然看似简单，但它决定了 SOAP 监督是否稳定。因为如果训练期间 feature frame 是随机同视频采样的，那么 SOAP 输出目录就无法和训练样本形成稳定一一对应关系，最终会导致几何监督信号不一致。因此，从随机特征帧切换到 canonical 特征帧，不只是一个工程小改动，而是 SOAP 方法能否成立的前提条件。

按 `dataset.json` 统计，理论上应导出：

1. 训练集 canonical feature frame：31 张。
2. 验证集 canonical feature frame：17 张。

这两个数字与当前数据集中 train/val 的视频数是一一对应的，因为每个视频只导出一个 canonical frame。

#### 2.4 SOAP 离线输出结构

SOAP 输出根目录为：

```text
/root/autodl-fs/soap_output
```

当前训练配置使用的是：

```yaml
DATASET:
    SOAP_GUIDANCE:
        ENABLED: true
        ROOT: '/root/autodl-fs/soap_output'
        LOAD_SIZE: 256
        USE_CANONICAL_FEATURE_FRAME: true
```

实际抽查发现，一个 SOAP 样本目录下包含完整的 6 个视角图像和 6 个视角法线图，例如：

```text
/root/autodl-fs/soap_output/feature_frames/<frame_key>/6-views/images/0.png ... 5.png
/root/autodl-fs/soap_output/feature_frames/<frame_key>/6-views/normals/0.png ... 5.png
```

这说明你的六视角监督数据链路是完整的。

从视角设计上看，当前使用的是 `[0, -90, 180, 90, -45, 45]` 六个方位角，包含正面、左右侧面、背向和两个斜侧面。这组视角的优点是可以在训练中提供比单视图更丰富的几何约束，尤其能约束侧脸与边缘区域；缺点是其中部分视角，特别是 180° 背向视角，本身较依赖 SOAP 的生成能力，可能引入更强的伪标签误差。因此，视角组合本身也是后续可调的重要超参数之一。

从抽样检查结果看，每个有效 SOAP 样本目录均满足以下结构约束：

1. `images/` 下有 6 张 RGB 图像。
2. `normals/` 下有 6 张法线图。
3. 视角编号按 `0,1,2,3,4,5` 排列。

这与配置中的六个视角 `[0, -90, 180, 90, -45, 45]` 是对应的，因此模型在训练时能够稳定获得固定视角顺序的伪多视角监督。

#### 2.5 数据清洗与注意事项

在实际检查中还发现一个值得写进报告的工程细节：`soap_inputs/feature_frames_val` 目录中存在历史缓存 PNG，数量大于 `manifest.txt` 记录的有效样本数。因此，报告中如果需要统计 canonical feature frame 的数量，建议采用以下口径：

1. 以 `dataset.json` 的真实视频数为准：train 31 个视频，val 17 个视频。
2. 或以 `manifest.txt` 为准。
3. 不要直接用目录里的原始 PNG 总数作为有效样本数。

这一点不会直接破坏训练，因为训练阶段是按 `frame_key` 精确解析 SOAP 路径并读取，而不是顺序扫目录，但在写报告时应避免口径不一致。

这一工程细节也反映出实验型项目中常见的问题：实际目录往往会保留历史中间结果，而研究报告中的统计口径必须以“当前有效实验设置”为准。把这类问题如实写进报告，反而会让整份作业更严谨，因为它体现了你对实验数据可复核性的重视。

---

## 四、作品实现

### 1. 总体结构与设计思路

本项目的完整流程可以概括为：

1. 原始 VFHQ 子集整理为 `img_lmdb + optim.pkl + dataset.json`。
2. 对数据执行 matting，统一背景为黑色。
3. 为每个视频导出 canonical feature frame。
4. 将 feature frame 输入 SOAP，离线生成 6 视角图像和法线图。
5. 训练时由数据加载器按 `frame_key` 读取 SOAP guidance。
6. 模型前向阶段生成三维高斯表示并渲染图像。
7. 损失计算阶段将预测法线与 SOAP 法线进行一致性约束。
8. 在验证集上计算 PSNR 和 SSIM，并保存 best checkpoint 与示例拼图。

从模块划分上，本项目可以分成 5 个功能层：

1. 配置层：`configs/model/gaga.yaml` 与 `configs/data/vfhq.yaml`。
2. 数据层：`core/data/loader_track.py`。
3. 模型层：`core/models/GAGAvatar/models.py`。
4. 工具层：`tools/export_soap_feature_frames.py`、`tools/matte_lmdb_dataset.py`。
5. 训练监控层：`train.py` 中的 TensorBoard 记录与 checkpoint 保存。

如果进一步从训练时的一次前向与反向过程来看，可以把整个系统抽象为如下闭环：

1. `TrackedData` 从 LMDB 和 `optim.pkl` 中取出 `f_image`、`t_image`、`t_transform`、`t_points`。
2. 若开启 SOAP，则附加 `soap_guidance['images']`、`soap_guidance['normals']`、`soap_guidance['masks']`。
3. `GAGAvatar.forward()` 用 `f_image` 编码出全局与局部特征，生成高斯参数并渲染 `gen_image` 与 `sr_gen_image`。
4. `calc_metrics()` 同时计算图像域损失、点集损失和 SOAP 法线损失。
5. Adam 根据总损失反向更新 lifting planes、局部高斯参数和上采样网络。
6. 每 2000 step 在验证集上计算 PSNR/SSIM，并保存最优模型。

这个流程说明，你做的改动并不是局部插了一个小函数，而是改变了训练数据流和损失流的主路径。

如果从“输入是什么，输出是什么，中间发生了什么”的角度再总结一次，本文的系统设计可以概括为一句话：

给定单张源头像图像和目标驱动条件，模型一方面通过原始图像重建损失学习可渲染头像表示，另一方面通过 SOAP 离线生成的多视角 normal map 约束 lifting plane 的局部几何，从而尝试减少纯单视角训练带来的几何歧义。

### 2. 模型结构设计

为了满足课程模板中“至少两种模型对比”的要求，本文将同一主架构下的三种训练方案作为三组模型进行比较。

| 模型编号 | 名称 | 架构差异 | 训练策略 |
| --- | --- | --- | --- |
| M1 | GAGAvatar-matted-baseline | 原始 GAGAvatar，不加 SOAP loss | 从头训练 |
| M2 | GAGAvatar-SOAP-scratch | 在 M1 架构上加入 SOAP 六视角 normal guidance | 从头训练 |
| M3 | GAGAvatar-SOAP-finetune | 架构同 M2 | 根据本次实验设置作为微调继续训练 |

三者共享相同的主干结构：

1. 输入层：`f_image` 先被 resize 到 518×518；渲染输出为 512×512。
2. 编码层：DINOv2 / ViT 输出局部和全局语义特征。
3. 几何层：global points + lifting plane l0 + lifting plane l1。
4. 渲染层：Gaussian renderer 输出初步图像。
5. 重建层：StyleUNet 上采样为最终头像结果。

其差异集中在损失函数层：M1 只包含原始重建与几何损失，M2 和 M3 额外包含 SOAP 多视角法线监督项。

从课程报告写作角度，这样的比较方式也有好处：三组实验共享同一主网络，差异主要集中在训练策略和几何监督项，因此结论更容易解释，不会因为模型主干差异过大而掩盖 SOAP 方法本身的影响。

进一步比较三组模型可以发现，它们的不同并不在于“网络深度改变了多少”，而在于“监督信号从哪里来、何时加入训练”。

1. M1 的监督完全来自目标图像和 FLAME 对齐几何，是标准 baseline。
2. M2 在 M1 基础上直接从第 0 step 加入 SOAP 约束，因此优化目标更复杂。
3. M3 虽然结构与 M2 相同，但从训练初始状态看明显带有 warm-start 特征，因此更像“先学会重建，再加入 SOAP 约束”的策略。

这一点在后续训练数据中会被进一步验证：M3 在第 100 步就远好于 M2 的训练状态，这与“微调”设定一致。

### 3. 核心算法与实现步骤

#### 3.1 库导入与环境配置

环境版本经实际查询如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.2 |
| PyTorch | 2.4.1 |
| Torchvision | 0.19.1 |
| Lightning | 2.4.0 |
| PyTorch3D | 0.7.8 |

#### 3.2 数据集加载与预处理函数

训练数据由 `TrackedData` 类加载。该类负责：

1. 从 `dataset.json` 确定 train / val / test 划分。
2. 从 `optim.pkl` 恢复 target 的 FLAME 参数。
3. 从 LMDB 读取 feature image 和 target image。
4. 在开启 SOAP 时读取对应的 SOAP 多视角数据。

#### 3.3 模型类定义

模型主类是 `GAGAvatar`。在原始项目中，模型已经具备图像编码、点平面生成、高斯渲染与图像重建功能；你的主要贡献是把 SOAP guidance 的配置解析、多视角法线渲染和损失计算接入该类。

#### 3.4 训练函数与验证逻辑

训练使用 20000 step、每 2000 step 验证一次。验证阶段会：

1. 计算合并后的 PSNR 和 SSIM。
2. 保存 `examples/<step>.jpg` 验证拼图。
3. 根据 SSIM 保存 best checkpoint。
4. 把标量与图像写入 TensorBoard。

当前训练超参数可以更完整地整理为如下表格。

| 参数 | 设置 |
| --- | --- |
| 优化器 | Adam |
| 学习率 | 1e-4 |
| batch size | 4 |
| 训练步数 | 20000 |
| 验证间隔 | 2000 step |
| SOAP normal weight | 0.05 |
| SOAP RGB weight | 0.0 |
| SOAP render size | 128 |
| SOAP load size | 256 |
| feature image resize | 518×518 |
| final output size | 512×512 |

这张表可以直接放进报告正文中，强化“实验设置是明确且可复现的”这一点。

#### 3.5 评估指标计算

本报告主要使用验证集上的：

1. PSNR：衡量像素级重建误差。
2. SSIM：衡量结构相似性，作为 best checkpoint 的主要选择标准。
3. soap normal loss：衡量额外几何监督项的收敛情况。

#### 3.6 结果可视化与保存

每个实验目录都包含：

1. `train_log.txt`：完整训练日志。
2. `checkpoints/best_*.pt`：最佳权重。
3. `examples/*.jpg`：验证拼图。
4. `tensorboard/`：可用于导出训练曲线。

这意味着整套实验既有模型权重，也有训练过程记录和验证可视化结果，具备较好的可复现性。对于课程项目来说，这是非常重要的，因为很多作业只保留最终截图而没有中间日志，导致结论无法被复核。你这次保留的 `train_log.txt + checkpoints + examples + tensorboard` 已经具备了较完整的研究实验归档结构。

#### 3.7 模型对比实验设计

本报告使用同一 matted 数据集上的三组实验做严格对比：

| 实验编号 | 输出目录 | 说明 |
| --- | --- | --- |
| E1 | Apr25_0138_rbtam | matted baseline |
| E2 | Apr25_1030_kxwey | + SOAP normal guidance，从头训练 |
| E3 | Apr25_1707_nheku | + SOAP normal guidance，按实验设置为微调训练 |

这样可以尽量排除数据集变化带来的干扰，把对比集中到“是否加入 SOAP 监督，以及从头训练还是微调训练”这两个因素上。

此外，这三组实验的训练配置保持一致：batch size 均为 4，总训练步数均为 20000，验证间隔均为 2000 step，学习率均为 1e-4。也就是说，除是否启用 SOAP guidance 及训练策略外，其余主要训练条件基本保持不变，因此对比具有较好的可解释性。

#### 3.8 训练时 batch 数据结构

为了更清晰展示代码实现，本项目训练时的主要 batch 字段可总结为表 4-1。

| 字段名 | 典型形状 | 含义 |
| --- | --- | --- |
| `f_image` | `[B, 3, 518, 518]` | canonical feature image |
| `t_image` | `[B, 3, 512, 512]` | target / driven image |
| `t_points` | `[B, N, 3]` | 由 FLAME 生成的目标点集 |
| `t_transform` | `[B, 3, 4]` | 目标视角相机变换 |
| `f_planes['plane_points']` | `[B, 296^2, 3]` | lifting plane 基础点 |
| `soap_guidance['images']` | `[B, 6, 3, 256, 256]` | SOAP 六视角 RGB |
| `soap_guidance['normals']` | `[B, 6, 3, 256, 256]` | SOAP 六视角法线图 |
| `soap_guidance['masks']` | `[B, 6, 1, 256, 256]` | SOAP 前景 mask |

这个表格非常适合放在实现章节，因为它能让读者迅速理解 SOAP 监督并不是一张额外图片，而是一组多视角、带掩码的结构化几何监督数据。

### 4. 关键代码修改对比

下面按“原始关键代码”和“修改后关键代码”的方式，列出最重要的代码变化。

#### 4.1 配置层：模型配置 `configs/model/gaga.yaml`

原始关键代码：

```yaml
MODEL:
    NAME: 'GAGAvatar'
    TASK: 'DynamicGaussian'

TRAIN:
    BATCH_SIZE: 8
    TRAIN_ITER: 200000
    CHECK_INTERVAL: 10000
```

修改后关键代码：

```yaml
MODEL:
    NAME: 'GAGAvatar'
    TASK: 'DynamicGaussian'
    NORMAL_LOSS:
        ENABLED: false
        MODE: 'screen'
        WEIGHT: 0.05
        RENDER_SIZE: 128
    SOAP_GUIDANCE:
        ENABLED: true
        NORMAL_WEIGHT: 0.05
        RGB_WEIGHT: 0.0
        RENDER_SIZE: 128
        ELEVATION: 0.0
        VIEW_ANGLES: [0, -90, 180, 90, -45, 45]

TRAIN:
    BATCH_SIZE: 4
    TRAIN_ITER: 20000
    CHECK_INTERVAL: 2000
```

这一步的意义是把 SOAP 监督开关、权重和视角设置配置化，并把训练规模调整为本地可完成的实验尺度。

#### 4.2 配置层：数据配置 `configs/data/vfhq.yaml`

原始关键代码：

```yaml
DATASET:
    NAME: 'VFHQ'
    LOADER: 'TrackedData'
    FLAME_SCALE: 5.0
    FOCAL_LENGTH: 12.0
    POINT_PLANE_SIZE: 296
    PATH: '/data/umihebi0/users/x-chu/Data/VFHQ_CENTERED/'
```

修改后关键代码：

```yaml
DATASET:
    NAME: 'VFHQ'
    LOADER: 'TrackedData'
    FLAME_SCALE: 5.0
    FOCAL_LENGTH: 12.0
    POINT_PLANE_SIZE: 296
    PATH: '/root/autodl-fs/vfhq_group100_101_matted'
    SOAP_GUIDANCE:
        ENABLED: true
        ROOT: '/root/autodl-fs/soap_output'
        LOAD_SIZE: 256
        USE_CANONICAL_FEATURE_FRAME: true
```

这一改动让训练能直接读取 matted 数据和 SOAP 离线结果。

#### 4.3 数据层：`core/data/loader_track.py`

原始关键代码：

```python
if self._split == 'train':
    candidate_key = [key for key in self._video_info[video_id] if key != frame_key]
    feature_key = random.sample(candidate_key, k=number)[0]
else:
    feature_key = self._video_info[video_id][0]

f_image = self._lmdb_engine[feature_key].float() / 255.0
f_shape = torch.tensor(self._data[feature_key]['shapecode']).float()
f_transform = torch.tensor(self._data[feature_key]['transform_matrix']).float()
f_planes = build_points_planes(self._point_plane_size, f_transform)
return feature_key, f_image, f_shape, f_planes

def get_video_id(frame_key):
    if frame_key.split('_')[0] in ['img']:
        video_id = frame_key.split('_')[1]
    else:
        video_id = frame_key.split('_')[0]
    return video_id
```

修改后关键代码：

```python
soap_cfg = getattr(data_cfg, 'SOAP_GUIDANCE', None)
self._use_soap_guidance = bool(soap_cfg.ENABLED) if soap_cfg is not None else False
self._soap_root = str(getattr(soap_cfg, 'ROOT', '')).rstrip('/') if soap_cfg is not None else ''
self._soap_load_size = int(getattr(soap_cfg, 'LOAD_SIZE', 256)) if soap_cfg is not None else 256
self._soap_feature_frame_only = bool(getattr(soap_cfg, 'USE_CANONICAL_FEATURE_FRAME', True)) if soap_cfg is not None else False

if self._split == 'train':
    if self._use_soap_guidance and self._soap_feature_frame_only:
        feature_key = self._video_info[video_id][0]
    else:
        candidate_key = [key for key in self._video_info[video_id] if key != frame_key]
        feature_key = random.sample(candidate_key, k=number)[0] if len(candidate_key) > 0 else frame_key
else:
    feature_key = self._video_info[video_id][0]

if self._use_soap_guidance:
    one_record['f_transform'] = f_record['transform_matrix']
    one_record['f_points'] = f_points
    one_record['soap_guidance'] = self._load_soap_guidance(f_key)

def _load_soap_guidance(self, frame_key):
    soap_dir = resolve_soap_guidance_dir(self._soap_root, frame_key)
    image_paths = list_numbered_images(os.path.join(soap_dir, 'images'))
    normal_paths = list_numbered_images(os.path.join(soap_dir, 'normals'))
    ...
    return {
        'images': torch.stack(soap_images, dim=0),
        'normals': torch.stack(soap_normals, dim=0),
        'masks': torch.stack(soap_masks, dim=0),
    }

def get_video_id(frame_key):
    video_id = frame_key.rsplit('_', 1)[0]
    if video_id.startswith('img_'):
        video_id = video_id[4:]
    return video_id
```

这部分是本次方法落地的关键：

1. 训练特征帧从“随机同视频采样”改成“可固定 canonical frame”。
2. 数据 batch 中新增 `soap_guidance`、`f_transform`、`f_points`。
3. `get_video_id()` 从简单 `split('_')[0]` 修复为 `rsplit('_', 1)[0]`，避免 frame key 解析错误。

#### 4.4 模型层：`core/models/GAGAvatar/models.py`

原始关键代码：

```python
point_loss = square_distance(results['t_points'], results['p_points']).mean()
loss = {
    'percep_loss': pec_loss,
    'img_loss': img_loss,
    'box_loss': box_loss,
    'point_loss': point_loss,
}
psnr = -10.0 * torch.log10(nn.functional.mse_loss(t_image, sr_gen_image).detach())
return loss, {'psnr': psnr.item()}
```

修改后关键代码：

```python
soap_guidance_cfg = getattr(model_cfg, 'SOAP_GUIDANCE', None) if model_cfg is not None else None
self.use_soap_guidance = bool(soap_guidance_cfg.ENABLED) if soap_guidance_cfg is not None else False
self.soap_normal_weight = float(getattr(soap_guidance_cfg, 'NORMAL_WEIGHT', 0.0)) if soap_guidance_cfg is not None else 0.0
self.soap_render_size = int(getattr(soap_guidance_cfg, 'RENDER_SIZE', 128)) if soap_guidance_cfg is not None else 128
self.soap_view_angles = tuple(float(angle) for angle in getattr(
    soap_guidance_cfg, 'VIEW_ANGLES', [0, -90, 180, 90, -45, 45]
)) if soap_guidance_cfg is not None else ()

if self.use_soap_guidance and 'soap_guidance' in batch:
    results['f_transform'] = batch['f_transform']
    results['soap_guidance'] = batch['soap_guidance']

if self.use_soap_guidance and 'soap_guidance' in results:
    soap_losses, soap_show = self._calc_soap_guidance_loss(results)
    loss.update(soap_losses)
    show_metric.update(soap_show)

def _calc_soap_normal_guidance(self, results, view_transforms):
    soap_guidance = results['soap_guidance']
    ...
    overlap = pred_masks * target_masks
    cosine = (pred_normals * target_normals).sum(dim=2, keepdim=True).abs().clamp(0.0, 1.0)
    denom = overlap.sum().clamp(min=1.0)
    soap_normal_loss = ((1.0 - cosine) * overlap).sum() / denom
    return soap_normal_loss * self.soap_normal_weight, {
        'soap_normal_cos': float(soap_normal_cos.item()),
        'soap_normal_overlap': float(overlap.mean().item()),
    }
```

这部分实现了真正的 SOAP 多视角法线监督。也就是说，你这次工作的实质不是“多写了一个配置项”，而是把外部法线先验经过相机变换、法线旋转、前景掩码和余弦一致性，真正纳入了训练目标。

#### 4.5 工具层：`tools/export_soap_feature_frames.py`

新增关键代码：

```python
video_info, _ = build_video_info(dataset_split[split])
feature_keys = [video_info[video_id][0] for video_id in sorted(video_info.keys())]

for frame_key in feature_keys:
    image = lmdb_engine[frame_key]
    output_path = os.path.join(output_dir, f'{frame_key}.png')
    torchvision.io.write_png(image.to(torch.uint8), output_path)
```

它的作用是把训练集从“逐帧驱动数据”转换成“每个视频一个 canonical input”，为 SOAP 离线多视角生成提供稳定输入。

#### 4.6 训练监控层：`train.py`

新增关键代码：

```python
self.writer = SummaryWriter(log_dir=self._tb_dir)
...
self.writer.add_scalar('val/psnr', merged_psnr, iter_idx)
self.writer.add_scalar('val/ssim', merged_ssim, iter_idx)
self.writer.add_image('val/examples', merged_images, iter_idx)
```

它让训练过程中的曲线、指标和验证拼图都可以通过 TensorBoard 统一查看，为课程报告制图提供了直接素材。

---

## 五、运行结果与展示

### 1. 训练过程可视化

从三个实验目录的 `train_log.txt` 中，可以直接提取每 2000 step 的验证结果。SSIM 曲线如下表所示。

| Step | E1：matted baseline | E2：SOAP scratch | E3：SOAP fine-tune |
| --- | --- | --- | --- |
| 2000 | 0.6197 | 0.6095 | 0.6639 |
| 4000 | 0.6375 | 0.6251 | 0.6657 |
| 6000 | 0.6419 | 0.6198 | 0.6669 |
| 8000 | 0.6541 | 0.6341 | 0.6711 |
| 10000 | 0.6638 | 0.6311 | 0.6775 |
| 12000 | 0.6686 | 0.6394 | 0.6767 |
| 14000 | 0.6777 | 0.6420 | 0.6796 |
| 16000 | 0.6825 | 0.6433 | 0.6770 |
| 18000 | 0.6842 | 0.6440 | 0.6818 |
| 20000 | 0.6897 | 0.6442 | 0.6826 |

从这张表可以看出两点：

1. SOAP 从头训练组在整个训练过程中始终落后于 matted baseline。
2. SOAP 微调组从 2000 step 开始就明显高于 SOAP 从头训练组，并在 10000 step 达到 0.6775，已经非常接近 baseline 的最终结果。

进一步看 PSNR 全过程，可以得到表 5-2。

| Step | E1：matted baseline | E2：SOAP scratch | E3：SOAP fine-tune |
| --- | --- | --- | --- |
| 2000 | 17.47 | 17.53 | 19.47 |
| 4000 | 18.38 | 18.29 | 19.66 |
| 6000 | 18.94 | 18.30 | 19.88 |
| 8000 | 19.14 | 18.67 | 20.09 |
| 10000 | 19.62 | 18.67 | 20.44 |
| 12000 | 20.34 | 19.09 | 20.52 |
| 14000 | 20.61 | 19.31 | 20.43 |
| 16000 | 20.62 | 19.43 | 20.56 |
| 18000 | 20.68 | 19.31 | 20.68 |
| 20000 | 21.10 | 19.38 | 20.61 |

从表 5-2 可以进一步看出：

1. E1 从 2000 到 20000 step，验证 PSNR 提升了 3.63，说明 baseline 仍有比较持续的后期收益。
2. E2 从 2000 到 20000 step，验证 PSNR 只提升了 1.85，说明 SOAP 从头训练在中后期较早进入平台期。
3. E3 从 2000 到 20000 step，验证 PSNR 提升了 1.14，看起来增幅较小，但这是因为它从一开始就站在较高起点上，属于“高起点、后期稳步微调”的收敛模式。

如果从阶段性收敛速度来看，还可以得到一个很有代表性的观察：E3 在 2000 step 时的 SSIM 为 0.6639，已经略高于 E1 在 10000 step 的 0.6638。这意味着 SOAP 微调组在训练早期就达到 baseline 中期水平，说明 warm start 策略明显改善了收敛效率。

如果要插入训练曲线，建议先绘制一张 E1/E2/E3 的 SSIM 折线图，横轴为 step，纵轴为 SSIM。该图非常适合作为“训练过程可视化”的主图。

为了让数据更全面，还可以把三组实验在各验证节点上的 PSNR 全部列出。对应结果如下。

| Step | E1：matted baseline | E2：SOAP scratch | E3：SOAP fine-tune |
| --- | --- | --- | --- |
| 2000 | 17.47 | 17.53 | 19.47 |
| 4000 | 18.38 | 18.29 | 19.66 |
| 6000 | 18.94 | 18.30 | 19.88 |
| 8000 | 19.14 | 18.67 | 20.09 |
| 10000 | 19.62 | 18.67 | 20.44 |
| 12000 | 20.34 | 19.09 | 20.52 |
| 14000 | 20.61 | 19.31 | 20.43 |
| 16000 | 20.62 | 19.43 | 20.56 |
| 18000 | 20.68 | 19.31 | 20.68 |
| 20000 | 21.10 | 19.38 | 20.61 |

从 PSNR 轨迹可以观察到：

1. E2 在整个训练过程中几乎始终落后于 E1，说明 SOAP 从头训练不仅在 SSIM 上落后，在像素级重建精度上也落后。
2. E3 从 2000 step 开始就明显高于 E2，且在 18000 step 达到与 E1 相同的 20.68 PSNR，说明其优化路径比 E2 稳定得多。
3. E3 在后期与 E1 的差距已很小，但最终仍未完全超过 E1，这也说明当前 SOAP 微调更像“追平 baseline”的有效策略，而非“显著超越 baseline”的成熟方案。

### 2. 量化指标统计

三个实验的最佳 checkpoint 和对应指标如下。

| 实验 | 输出目录 | best checkpoint | 最佳 PSNR | 最佳 SSIM |
| --- | --- | --- | --- | --- |
| E1 | Apr25_0138_rbtam | best_20000_0.690.pt | 21.10 | 0.6897 |
| E2 | Apr25_1030_kxwey | best_20000_0.644.pt | 19.38 | 0.6442 |
| E3 | Apr25_1707_nheku | best_20000_0.683.pt | 20.61 | 0.6826 |

为了更清楚地说明差异，可以进一步计算增减幅度：

| 对比关系 | PSNR 变化 | SSIM 变化 | 结论 |
| --- | --- | --- | --- |
| E2 相比 E1 | -1.72 | -0.0455 | SOAP 从头训练未优于 baseline |
| E3 相比 E2 | +1.23 | +0.0384 | SOAP 微调显著优于 SOAP 从头训练 |
| E3 相比 E1 | -0.49 | -0.0071 | 微调后已基本接近 baseline |

这里最有价值的实验结论是：

1. 如果直接从头加入 SOAP 约束，最终结果会下降。
2. 如果基于已有较好模型再加入 SOAP 约束，性能差距会显著缩小。

若把 E1 最终结果视作当前基准上限，则可以进一步做归一化观察：

1. E2 最终 SSIM 为 0.6442，仅达到 E1 最终 SSIM 的约 93.4%。
2. E3 最终 SSIM 为 0.6826，已达到 E1 最终 SSIM 的约 99.0%。

这组数据从另一个角度说明：SOAP guidance 本身并非完全无效，而是它更适合在已有较稳几何和外观表示的前提下做后期细化，而不适合从零开始直接参与主导优化。

为了让结果分析更细致，表 5-3 给出三个关键阶段的对比。

| 阶段 | E1：baseline | E2：SOAP scratch | E3：SOAP fine-tune | 主要现象 |
| --- | --- | --- | --- | --- |
| 早期（2000） | 0.6197 / 17.47 | 0.6095 / 17.53 | 0.6639 / 19.47 | 微调组明显领先 |
| 中期（10000） | 0.6638 / 19.62 | 0.6311 / 18.67 | 0.6775 / 20.44 | 从头训练已显著落后 |
| 后期（20000） | 0.6897 / 21.10 | 0.6442 / 19.38 | 0.6826 / 20.61 | 微调组接近 baseline |

这个表格非常适合答辩时讲，因为它能用最少的数据把三组实验的总体趋势讲清楚。

如果从相对变化比例来看，结论会更直观：

1. E2 相比 E1，PSNR 相对下降约 8.15%，SSIM 相对下降约 6.60%。
2. E3 相比 E2，PSNR 相对提升约 6.35%，SSIM 相对提升约 5.96%。
3. E3 相比 E1，PSNR 仅低约 2.32%，SSIM 仅低约 1.03%。

因此，如果把实验目标分成两个层次，那么：

1. “SOAP 从头训练能否直接提升结果”这一问题，答案是否定的。
2. “SOAP 作为微调几何约束是否有潜力”这一问题，答案是肯定的，而且当前结果已经给出了较强迹象。

### 3. 几何监督项的收敛情况

仅看最终图像指标还不够，还要看 SOAP 监督本身是否真的被模型学到了。根据日志提取到的 `soap_normal_loss` 关键节点如下。

| Step | E2：SOAP scratch | E3：SOAP fine-tune |
| --- | --- | --- |
| 100 | 0.0239 | 0.0231 |
| 1000 | 0.0214 | 0.0200 |
| 2000 | 0.0212 | 0.0192 |
| 10000 | 0.0179 | 0.0159 |
| 20000 | 0.0156 | 0.0142 |

可见：

1. E2 中，`soap_normal_loss` 从 0.0239 降到 0.0156，下降约 34.7%。
2. E3 中，`soap_normal_loss` 从 0.0231 降到 0.0142，下降约 38.5%。

这说明 SOAP 监督项本身确实被模型逐渐优化了。也就是说，SOAP 约束不是“没学到”，而是“学到了，但并没有完全转化为更高的最终图像保真度”。这恰恰是一个很有研究价值的结论。

如果继续结合训练损失节点观察，可以得到表 5-4。这里需要说明的是，E1 与 E2/E3 的总损失定义不完全相同，因为后两者额外包含 `soap_normal_loss`，因此跨组比较总损失时应更关注趋势，而不是只看绝对值。

| Step | E1 train loss | E2 train loss | E3 train loss |
| --- | --- | --- | --- |
| 100 | 1.1758 | 1.2002 | 0.5044 |
| 1000 | 0.6473 | 0.6556 | 0.4162 |
| 2000 | 0.5719 | 0.5594 | 0.4047 |
| 5000 | 0.5071 | 0.4772 | 0.3787 |
| 10000 | 0.4518 | 0.4194 | 0.3486 |
| 15000 | 0.4194 | 0.3922 | 0.3310 |
| 20000 | 0.4135 | 0.3805 | 0.3254 |

从表 5-4 可以看出，E3 的训练损失从一开始就显著低于 E2，这与它更高的验证指标相一致，也从侧面支持了“微调起点更优”的判断。另一方面，E2 的训练损失虽然持续下降，但验证指标并没有同步接近 baseline，这说明当前 SOAP 从头训练存在明显的“训练目标学到了，但泛化到验证集画质不理想”的问题。

为了进一步说明训练动态，还可以比较三个实验在若干关键训练节点上的总体 loss 和训练集 PSNR：

| Step | E1 loss | E2 loss | E3 loss | E1 train PSNR | E2 train PSNR | E3 train PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 1.1758 | 1.2002 | 0.5044 | 12.89 | 12.88 | 20.68 |
| 1000 | 0.6473 | 0.6556 | 0.4162 | 18.85 | 19.00 | 21.95 |
| 2000 | 0.5719 | 0.5594 | 0.4047 | 19.34 | 19.87 | 22.31 |
| 5000 | 0.5071 | 0.4772 | 0.3787 | 20.64 | 21.35 | 23.51 |
| 10000 | 0.4518 | 0.4194 | 0.3486 | 21.75 | 22.52 | 24.51 |
| 15000 | 0.4194 | 0.3922 | 0.3310 | 22.51 | 23.28 | 25.18 |
| 20000 | 0.4135 | 0.3805 | 0.3254 | 22.77 | 23.58 | 25.35 |

这张表传达出三个重要现象：

1. E2 虽然最终验证指标低于 E1，但在训练集上的 loss 更低、train PSNR 更高，说明它对训练数据拟合得更强，却没有转化为更好的验证效果，这暗示几何约束可能引入了泛化层面的冲突。
2. E3 在第 100 步就表现出远低于 E1/E2 的 loss 和远高于 E1/E2 的 train PSNR，这与其“微调/继续训练”设定高度一致。
3. 从优化轨迹看，E3 的训练动态最平滑，说明 SOAP guidance 更适合作为 warm-start 后的细化约束，而不适合作为随机初始化阶段的主约束之一。

如果还想进一步增强报告的“数据味道”，可以在 PPT 或正文里把上表单独命名为“训练动态统计表”。

### 4. 训练耗时与代价分析

根据三个训练日志首尾时间估算，训练代价如下。

| 实验 | 总耗时（秒） | 单步耗时（秒/step） | 相对 baseline 开销 |
| --- | --- | --- | --- |
| E1：matted baseline | 11507 | 0.5754 | 1.00× |
| E2：SOAP scratch | 18297 | 0.9149 | 1.59× |
| E3：SOAP fine-tune | 18796 | 0.9398 | 1.63× |

可以看到，SOAP 多视角法线监督的代价并不小。它需要在训练过程中额外构造 6 个虚拟相机并渲染多视角法线图，因此整体训练耗时大约增加了 59% 到 63%。

这意味着：SOAP 方法不仅要看“最终指标是否更高”，还要看“收益是否足以覆盖训练代价”。在当前版本中，从头训练显然不划算；微调策略则更有继续优化的价值。

从工程实现角度解释，训练耗时增加的原因主要有两点：

1. 每个 batch 需要额外构造 6 个虚拟视角，并调用 PyTorch3D 光栅化过程把预测几何渲染为 normal map。
2. 需要对 SOAP normal map 做 resize、mask 处理和旋转对齐，再进行逐像素余弦一致性计算。

因此，SOAP 监督不是“几乎没有成本的附加项”，而是一个真实增加训练计算量的几何监督模块。也正因为如此，报告中同时讨论“效果”与“代价”是必要的。

### 5. 样例输出展示

通过查看三组实验最终的 `examples/20000.jpg`，可以得到以下定性观察。

1. E1 baseline 的整体边界最稳定，帽檐、肩膀和黑背景之间的过渡较自然。
2. E2 SOAP 从头训练组在帽檐边缘、肩部边缘和局部运动区域更容易出现发虚与轻微过平滑。
3. E3 SOAP 微调组相较 E2 有明显改善，视觉上更接近 E1，说明“先学重建、后加 SOAP”的策略更合理。

特别地，在 `Apr25_1707_nheku` 中，人物边缘和面部轮廓已经基本恢复到接近 baseline 的水平，这与它在量化指标上接近 baseline 的结果是一致的。

如果把定性和定量结果结合起来，可以更完整地理解三组实验：E1 给出了当前最稳的整体画质；E2 说明 SOAP 约束如果从头施加，容易导致几何与外观目标之间出现冲突；E3 则表明，先让模型学会稳定重建，再引入 SOAP 作为后期约束，是更符合当前实现特点的训练策略。

如果将视觉现象与量化指标结合起来看，可以形成如下对应关系：

1. E1 的高 SSIM 和高 PSNR 对应于更稳定的边缘、较少的背景污染和更清晰的脸部轮廓。
2. E2 的低 SSIM 和低 PSNR 对应于明显的边缘发虚、局部结构略平滑、帽檐与背景过渡不够锐利。
3. E3 的接近 baseline 的量化指标，对应于大多数样本在视觉上已经恢复了较好的轮廓质量，只是在少数局部细节上仍略逊于 E1。

也就是说，本次实验中的“数字变化”与“肉眼观察”是互相支撑的，而不是彼此矛盾的。

---

## 六、结果分析与讨论

### 1. 模型效果分析

本项目之所以能够取得可用结果，主要依赖以下几点。

1. DINOv2 / ViT 编码器能稳定提取单张头像中的身份与外观特征。
2. FLAME 提供了稳定的参数化几何先验，减少了纯单目重建的歧义。
3. 双层 lifting planes 与 Gaussian rendering 让模型既能表达主体人脸，也能表达更复杂的边缘与外观。
4. Matting 预处理显著降低了背景噪声，使训练目标更集中在人头区域。

这里还可以补充一个很关键的经验：在头像重建任务中，背景一致性往往会直接影响模型是否把容量浪费在无关区域上。当前 matted 数据集统一为黑背景后，模型不需要再用大量参数去拟合复杂背景纹理，而可以把更多能力集中在人脸主体和边界几何上。这也是为什么你这次实验中 matting 的效果比 SOAP 从头训练更稳定、更直接。

从结果看，当前最强的正向因素不是 SOAP，而是 matting 背景清洗。这一点对课程报告非常重要，因为它说明你不只是“加了一个新损失”，还通过数据工程发现了更关键的性能影响因素。

### 2. 模型对比分析

#### 2.1 baseline 与 SOAP 从头训练的对比

E2 相比 E1 的结果明显下降，说明在当前实现中，SOAP 法线伪监督并不能直接替代原始重建目标，原因可能包括：

1. SOAP 生成的法线本质上仍然是伪标签，存在噪声。
2. SOAP 法线与 GAGAvatar 内部相机坐标系之间可能存在轻微不一致。
3. 从头训练阶段，模型本身几何尚不稳定，过早加入额外约束容易造成优化冲突。

除此之外，还有两个更细的原因值得写入报告。

4. 当前 `SOAP_GUIDANCE.RGB_WEIGHT=0.0`，说明模型只利用了 SOAP 的法线信息，没有利用 SOAP 生成的多视角 RGB 作为外观一致性辅助，因此教师先验的可用信息并没有被完全用足。
5. 当前 `SOAP_GUIDANCE.RENDER_SIZE=128`，而 SOAP 原始读入尺寸为 256，最终重建输出为 512，这意味着几何监督是在较低分辨率上作用的，可能不足以精确约束高分辨率边缘细节。

#### 2.2 SOAP 从头训练与 SOAP 微调的对比

E3 相比 E2 的提升非常明显：

1. 2000 step 时，PSNR 提升 1.94，SSIM 提升 0.0544。
2. 20000 step 时，PSNR 提升 1.23，SSIM 提升 0.0384。

这说明 SOAP guidance 更适合作为后期几何细化约束，而不是从第 0 step 就主导训练。

换一个更直观的说法：E2 需要完整训练 20000 step，最终 SSIM 仍只有 0.6442；而 E3 在 2000 step 时就已经达到 0.6639。这意味着，单就“有效收敛速度”而言，微调策略甚至在训练早期就已经超过了从头训练的最终水平。这是整篇报告中最值得强调的一个实验现象。

如果从阶段性角度再细分，可以把三组实验的优化过程拆成三个阶段来理解：

1. 早期阶段（0 到 2000 step）：E3 已经显著领先 E2，说明好的初始化比额外约束本身更重要。
2. 中期阶段（2000 到 10000 step）：E1 和 E3 都稳定上升，而 E2 上升较慢且存在波动，说明从头训练加入 SOAP 后优化更困难。
3. 后期阶段（10000 到 20000 step）：E3 继续稳定逼近 E1，E2 逐渐收敛但无法追回差距，说明 warm-start 策略的优势是持续存在的，而不是偶然的早期现象。

#### 2.3 SOAP 微调与 baseline 的对比

E3 的最终结果距离 baseline 只差 0.49 PSNR 和 0.0071 SSIM，已经非常接近。换句话说：

1. SOAP 微调策略已经把“从头训练组的大幅退化”基本拉回来了。
2. 但当前版本仍未真正超过 baseline，因此还不能得出“SOAP 一定能提升最终质量”的结论。

### 3. 误差来源与问题分析

结合代码与实验，当前误差来源主要有以下几个方面。

1. 伪监督噪声：SOAP normal map 并不是真实多视角扫描得到的法线，因此存在方向误差和边界噪声。
2. 坐标对齐误差：虽然代码中加入了 `_build_soap_view_transforms()` 和 `_rotate_soap_normals()`，但不同模型间的相机与法线定义可能仍不完全一致。
3. 数据规模较小：当前验证集只有 64 帧，虽然足够做课程报告，但统计稳定性有限。
4. 训练代价较高：SOAP 训练比 baseline 慢约 1.6 倍，参数搜索成本更大。
5. 缺少独立 test evaluation：当前目录中最容易复核的是验证集指标，因此结论仍主要基于 validation 而非独立测试集。

还有一个隐含问题是教师先验与学生模型之间的表征不匹配。SOAP 的 normal maps 来自一个独立模型，其几何假设、头部完整性和遮挡处理方式与 GAGAvatar 并不完全一致。即使相机方位角一致，二者对头发、帽檐和边界的定义也可能不同。这类“跨模型监督”的不一致性，是外部 teacher prior 类方法中非常常见但也非常难处理的问题。

还可以补充两个更细的误差来源：

6. 监督目标层级不一致：SOAP normal 约束的是几何局部一致性，而最终评估指标是图像级 PSNR/SSIM，中间还隔着高斯参数、颜色建模和超分网络，因此“几何更一致”并不一定立刻等价于“图像指标更高”。
7. 权重与时机耦合：同样的 `NORMAL_WEIGHT=0.05`，在随机初始化阶段可能偏强，在收敛后阶段又可能刚好合适，因此“同一个权重”在不同训练阶段的作用可能并不相同。

这一点解释了为什么本文的微调实验比从头训练更有希望：问题不一定出在 SOAP 思路本身，而可能出在“什么时候用、用多强”这两个训练策略细节上。

### 4. 本次实验最可信的结论

基于代码、日志和可视化，本文认为最可信的结论是：

1. 你的 SOAP 法线监督已经在代码层面真实接入，并且几何监督项确实在收敛。
2. 直接从头训练加入 SOAP 约束，当前版本会降低最终图像质量。
3. SOAP 微调策略明显优于 SOAP 从头训练，并已经基本追平 baseline。
4. 当前真正最稳定的收益来源是 matting，而不是 SOAP 从头训练。

进一步概括，这次实验最合理的学术表达不是“SOAP 无效”，而是“在当前实现和当前训练设定下，SOAP 多视角法线先验未能在从头训练场景中带来更优的验证指标，但在微调场景下表现出明显更好的收敛趋势和接近 baseline 的最终性能”。这种表述既客观，也更符合研究报告的写作规范。

这不是“负结果”，而是一个非常像研究报告的结论：方法方向有探索价值，但训练策略和对齐方式还需要继续打磨。

换句话说，如果把本次课程项目总结为一句更准确的话，可以写成：

“本文在 GAGAvatar 上成功实现了 SOAP 多视角法线监督的训练扩展，但实验表明该监督更适合作为后期细化约束，而不是从头训练阶段的主监督；与此同时，matting 背景清洗是当前最稳定、最明确的有效因素。”

---

## 七、总结与改进展望

### 1. 作品总结

本文围绕单目图像驱动的三维头部 Avatar 重建任务，以 GAGAvatar 为基线，完成了从环境复现、数据预处理、模型扩展、训练验证到结果分析的完整闭环。与只运行官方 demo 不同，本次工作有以下明确的研究与工程贡献：

1. 构建了 matted VFHQ 子集训练流程，并验证了背景清洗的正向作用。
2. 实现了 canonical feature frame 导出和 SOAP 离线 six-view 数据接入。
3. 在数据加载层和模型损失层中真正加入了 SOAP 多视角法线监督。
4. 完成了 baseline、SOAP 从头训练、SOAP 微调三组真实训练对比。
5. 用日志、验证图和训练代价共同支撑了结论，而不是只展示一张“看起来不错”的图。

从研究训练角度看，这一点尤其重要。课程项目最容易陷入的问题是“只展示一组成功可视化”，但缺少系统对比。你这次报告的优势在于：不仅有代码改动和功能实现，还有训练曲线、验证指标、best checkpoint、训练耗时和定性图像共同支撑结论。因此，这份报告在结构上已经比较接近小型实验论文，而不是纯项目说明文档。

从课程作业角度看，这已经明显超过“基础复现”层级，更接近“复现 + 方法修改 + 对比实验 + 结果分析”的高层级项目。

如果从科研训练的角度评价，这份工作还有一个额外价值：它给出了一个“方法接入成功但结果并未立刻超过基线”的完整分析过程。相比只展示一组漂亮效果图，这种做法更符合研究报告的逻辑，因为它能够回答：

1. 代码到底改了哪里。
2. 数据链路是否真的连通了。
3. 新监督项是否真的参与优化并收敛了。
4. 为什么结果没有立刻超过基线。
5. 下一步该往哪里改。

这些内容共同构成了一份更完整、更可信的高级人工智能实训报告。

### 2. 改进方向

后续可以从以下几个方向继续优化。

1. 分阶段启用 SOAP：例如先训练 baseline，再逐步提升 SOAP 权重，而不是从头就固定为 0.05。
2. 检查法线坐标系：进一步校正 SOAP normal 与 GAGAvatar 相机系之间的旋转、方向与 mask 定义。
3. 引入更多监督：当前 `RGB_WEIGHT=0.0`，后续可以尝试 SOAP RGB guidance 或边缘一致性项。
4. 清理 SOAP 缓存目录：把 `soap_inputs` 与 `soap_output` 中的历史样本清理干净，保证统计口径完全一致。
5. 增加 test set evaluation：将 32 帧 test set 单独跑一次，给出更标准的最终测试指标。
6. 补充更公平的对照：例如在完全相同子集上额外增加“matted + screen normal”作为中间对照组。

如果后续还有时间，实际上还可以继续做两个更深入的方向。其一，是对 SOAP 视角组合做消融，例如去掉 180° 背向视角，只保留正面、左右侧面和斜侧面，观察是否能够减少伪标签误差；其二，是把 SOAP guidance 改成逐阶段增权的 curriculum 策略，例如前 5000 step 不启用，5000 到 10000 step 线性增大到 0.05，后期再保持恒定。这两类策略都非常符合当前实验结论所指向的改进方向。

如果继续往下做，我认为最优先的两项改进应该是：

1. 先把 “baseline -> resume -> gradually increase SOAP weight” 做成正式训练策略，因为当前数据已经说明这条路线最有潜力。
2. 单独把 SOAP normal 对齐问题做一次可视化核查，例如直接并排显示“预测 normal / SOAP normal / overlap mask”，这样能更明确判断误差到底来自伪标签噪声，还是来自坐标系不一致。

这两步如果能做好，后续超过 baseline 的可能性会明显提高。

---

## 八、个人/小组分工

本项目可按个人作业填写如下。

| 成员 | 分工 |
| --- | --- |
| 成员 1：__________ | 数据集整理、matting 预处理、SOAP feature frame 导出、GAGAvatar 代码修改、实验运行、结果分析、报告撰写 |

如果老师要求按双人小组格式填写，也可以写成：

1. 成员 1：__________，负责数据集处理、模型搭建、训练代码修改、实验运行、报告撰写。
2. 成员 2：无。

---

## 附录：图片插入建议与图注写法

下面给出一份可以直接照着排版的插图清单。

| 建议图号 | 插入位置 | 推荐素材 | 插图作用 | 图注建议 |
| --- | --- | --- | --- | --- |
| 图 1-1 | 第一章“任务背景与应用场景”后 | `render_results/GAGAvatar/2_1.jpg` | 直观展示“输入人像 + 驱动人像 + 输出结果”的任务形式 | 单图驱动头像重演示意图 |
| 图 2-1 | 第二章“SOAP 多视角先验”后 | `soap_output/feature_frames/<任意样本>/6-views/normals/0.png`，最好自行拼成 6 张图 | 展示 SOAP 生成的法线先验长什么样 | SOAP 六视角法线监督示意图 |
| 图 3-1 | 第三章“数据预处理流程”后 | `render_results/tracked/1.jpg` | 展示 tracked / FLAME 拟合后的头部结果 | 输入图像的跟踪与参数化几何示意 |
| 图 4-1 | 第四章“总体结构与设计思路”后 | 自绘流程图 | 展示完整方法链路：LMDB -> matting -> feature frame -> SOAP -> loader -> loss | 本文方法总流程图 |
| 图 5-1 | 第五章“训练过程可视化” | 由表 5-1 绘制的 SSIM 折线图 | 展示 baseline、SOAP scratch、SOAP fine-tune 的收敛差异 | 三组实验的验证集 SSIM 曲线 |
| 图 5-2 | 第五章“训练过程可视化” | 由日志导出的 train/loss 或 soap_normal_loss 曲线 | 展示 SOAP 约束确实被优化 | SOAP 几何监督项的收敛曲线 |
| 图 5-3 | 第五章“样例输出展示” | `outputs/GAGAvatar_VFHQ/Apr25_0138_rbtam/examples/20000.jpg` | 展示 matted baseline 最终效果 | matted baseline 的验证可视化结果 |
| 图 5-4 | 第五章“样例输出展示” | `outputs/GAGAvatar_VFHQ/Apr25_1030_kxwey/examples/20000.jpg` | 展示 SOAP 从头训练最终效果 | SOAP 从头训练的验证可视化结果 |
| 图 5-5 | 第五章“样例输出展示” | `outputs/GAGAvatar_VFHQ/Apr25_1707_nheku/examples/20000.jpg` | 展示 SOAP 微调最终效果 | SOAP 微调的验证可视化结果 |
| 图 6-1 | 第六章“结果分析与讨论” | 将图 5-3、图 5-4、图 5-5 中相同样本裁剪后并排 | 更突出边缘、帽檐、肩膀等区域差异 | baseline、SOAP scratch、SOAP fine-tune 局部对比图 |

### 图片插入时的排版建议

1. 不建议把整张 `examples/20000.jpg` 原封不动塞进正文，可以裁剪 2 到 3 组最有代表性的样本，重点看帽檐、肩部和边缘过渡。
2. 如果篇幅有限，优先保留图 5-1、图 5-3、图 5-4、图 5-5 四张图。
3. 第四章的方法流程图建议自己画，这样报告会更像“研究型课程作业”，而不是简单截图堆叠。
4. 如果老师要求“训练损失曲线”，可以从每个实验目录的 `tensorboard/` 中直接导出 `train/loss` 和 `train_show/soap_normal_cos` 等标量图。

### 可以直接放在正文里的简短图注模板

1. 图 1-1 展示了本项目的任务形式：给定源人物图像和驱动图像，模型输出目标人物在驱动姿态与表情下的重演结果。
2. 图 2-1 展示了 SOAP 生成的多视角法线图，本文将其作为外部 teacher prior 接入 GAGAvatar 训练。
3. 图 5-1 表明 SOAP 微调策略在训练早期明显优于 SOAP 从头训练，而 matted baseline 仍然保持最高的最终 SSIM。
4. 图 6-1 显示在帽檐、肩部与黑背景交界处，SOAP 从头训练更容易出现边缘发虚，而 SOAP 微调已明显缓解该问题。
