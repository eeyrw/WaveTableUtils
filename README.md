# WaveTableUtils

从 SoundFont2 (`.sf2`) 文件中提取波形样本，经过重采样、FFT 带限、循环点修正、增益调整等信号处理，生成面向嵌入式 MCU（8051、AVR、STM8 等）的 C / Python 波表代码。

## 依赖

```bash
pip install -r requirements.txt
```

## 用法

```bash
# 列出 SF2 中的所有采样
python WaveTableGenerator.py --sf2 soundfont/MusicBox.sf2 --listSf2

# 生成特定采样的波表代码
python WaveTableGenerator.py --sf2 soundfont/MusicBox.sf2 --template 8051_sdcc \
    --sampleName "Square Wave C5" --outSampleRate 32000 --outSampleWidth 1 \
    --lowestNote 36 --padding --outputDir ./out

# 生成频谱/时域/循环点分析 PDF
python WaveTableGenerator.py --sf2 soundfont/MusicBox.sf2 --template generic \
    --sampleName "Music Box C5" --spectrumPdf analysis.pdf

# 生成 10-bit 音频到 16-bit PWM 的查找表 C 数组
python Mapping.py
```

## 模板

内置模板位于 `template/` 目录：

- `8051_sdcc` — SDCC 8051 汇编/C
- `avr_gcc` — AVR-GCC
- `stm8_sdcc` — STM8 SDCC
- `generic` — 通用 C
- `python` — Python

可通过 `--extraTemplate` 传入自定义模板文件。

## 处理流程

1. 从 SF2 提取 attack + loop 采样
2. 重采样/声道转换/位宽转换到目标格式
3. FFT 估计基频
4. FFT 带限 — 防止在最低音符（`--lowestNote`，默认 MIDI 36）上混叠
5. 自动修正循环点不连续性（最小化幅值 + 斜率误差）
6. 增益调整（`--gainDb`，如 -6.0 dB）
7. 渲染模板文件（样本数据 + 预计算增量表）
8. 可选输出频谱 PDF（`--spectrumPdf`）

## 许可证

LGPL-3.0
