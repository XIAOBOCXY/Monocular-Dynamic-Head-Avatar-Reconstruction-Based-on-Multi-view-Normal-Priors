## 基于多视角法线先验的单目动态头部 Avatar 重建

摘要：随着三维高斯泼溅（3DGS）技术在神经渲染领域的兴起，单目动态头部 Avatar 重建在虚拟数字人领域展现出巨大潜力。然而，单目输入存在的深度歧义和遮挡问题，导致模型在处理侧脸及复杂边缘几何时易出现坍塌或伪影。本文以 GAGAvatar 为基线模型，引入 SOAP（Style-Omniscient Animatable Portraits）生成的多视角法线先验作为额外几何监督信号，旨在增强重建 Avatar 的空间一致性。本文首先构建了基于 VFHQ 数据集的 Matting 预处理流程，并设计了标准特征帧导出与多视角先验生成方案。实验对比了基线模型、SOAP 从头训练以及 SOAP 微调三种策略。结果表明，直接引入 SOAP 先验进行从头训练会导致图像质量下降，但采用微调策略能有效收敛几何损失，并在保证图像质量的同时增强局部几何的稳定性。本文的研究为结合离线生成先验辅助在线三维重建提供了技术参考。

关键词：单目头部重建；多视角法线先验；神经渲染；3D Gaussian Splatting
###  参考文献
[1]Chu X, Harada T. Generalizable and animatable gaussian head avatar[J]. Advances in Neural Information Processing Systems, 2024, 37: 57642-57670.

[2]Kerbl B, Kopanas G, Leimkühler T, et al. 3d gaussian splatting for real-time radiance field rendering[J]. ACM Trans. Graph., 2023, 42(4): 139:1-139:14.

[3]Liao T, Zheng Y, Xiu Y, et al. SOAP: Style-Omniscient Animatable Portraits[C]//Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 2025: 1-11.

[4]Wang L X, Zhang H, Dong C, et al. VFHQ: A High-Quality Dataset and Benchmark for Video Face Super-Resolution[J]. arXiv preprint arXiv:2205.03409, 2022.

[5]Oquab M, Darcet T, Moutakanni T, et al. Dinov2: Learning robust visual features without supervision[J]. arXiv preprint arXiv:2304.07193, 2023.

[6]Li T, Bolkart T, Black M J, et al. Learning a model of facial shape and expression from 4D scans[J]. ACM Trans. Graph., 2017, 36(6): 194:1-194:17.

[7]Wang L, Zhao X, Sun J, et al. Styleavatar: Real-time photo-realistic portrait avatar from a single video[C]//ACM SIGGRAPH 2023 Conference Proceedings. 2023: 1-10.