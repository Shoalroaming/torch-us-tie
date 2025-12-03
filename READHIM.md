# 光强传输方程的迭代解法

基于PyTorch实现的迭代TIE（Transport of Intensity Equation）求解算法，算法源自论文[*On a universal solution to the transport-of-intensity equation*](https://doi.org/10.1364/OL.391823)。

[![arXiv](https://img.shields.io/badge/arXiv-1912.07371-B31B1B.svg?logo=arxiv)](https://arxiv.org/abs/1912.07371)
[![DOI](https://img.shields.io/badge/DOI-10.1364/OL.391823-blue)](https://doi.org/10.1364/OL.391823)

## 功能特性

- TIE迭代求解
- 多种早停策略
- GPU加速支持
- 相位重建质量评估

## 文件结构

```
.
├─ asp.py              # 角谱法
├─ dog=19.9mm.tif      # 19.9mm处灰度图
├─ dog=20mm.tif        # 20mm处灰度图
├─ dog=20.1mm.tif      # 20.1mm处灰度图
├─ test_dog.py         # 实际图像测试
└─ us_tie.py           # 通用TIE求解器
```

## 核心函数

### us_fft_tie

```python
us_fft_tie(
    I_0: torch.Tensor,              # 聚焦平面的二维光强分布 (H, W)
    dIdz: torch.Tensor,             # 光强在z方向的导数 (H, W)
    lam: float,                     # 光的波长
    dx: float,                      # x方向空间采样间隔
    dy: float,                      # y方向空间采样间隔
    max_iter: int = 50,             # 最大迭代次数
    tol: float = 1e-3,              # 收敛容差
    st: float = 0.95,               # 收敛停滞阈值
    device: torch.device = 'cuda'   # 计算设备
) -> torch.Tensor                   # 返回恢复的相位分布
```

## 使用示例

### 相位恢复演示

```bash
python test_dog.py
```

加载离焦图像，计算光强导数，使用单次和迭代TIE方法恢复相位，并通过角谱法验证重建质量。

## 输入输出

### 输入：
- 光强图和其对应的光强导数
- 波长、像素尺寸、等物理参数

### 输出：
- 恢复的相位图

## 依赖

- PyTorch
- NumPy
- PIL
- matplotlib（仅测试脚本）

## 引用
如果您在研究中使用了这段代码~~（当然，这不太可能）~~，请引用原论文：
- **BibTeX:**
  ```bibtex
  @article{Zhang:20,
    author={Jialin Zhang and Qian Chen and Jiasong Sun and Long Tian and Chao Zuo},
    journal = {Opt. Lett.},
    keywords = {Fourier transforms; Imaging systems; Microlens arrays; Optical fields; Phase imaging; Phase retrieval},
    number = {13},
    pages = {3649--3652},
    publisher = {Optica Publishing Group},
    title = {On a universal solution to the transport-of-intensity equation},
    volume = {45},
    month = {Jul},
    year = {2020},
    url = {https://opg.optica.org/ol/abstract.cfm?URI=ol-45-13-3649},
    doi = {10.1364/OL.391823},
}