# ICAT_Diversity-Driven Visual Jailbreaking: A Quality-Diversity Approach with Stagnation Recovery

**An adaptive, query-efficient black-box framework for evaluating Vision-Language Model safety through evolutionary optimization with stagnation recovery mechanisms and hybrid attack vectors.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🔬 Research Objective

This framework evaluates the **safety robustness** of Vision-Language Models (Qwen2-VL) using **pure black-box attacks** with Quality-Diversity optimization. The system combines CMA-ME evolution with advanced prompt injection techniques.

### Key Features

- **Pure Black-Box**: No gradient access - only text output feedback
- **QD Optimization**: CMA-ME with 10×10 behavioral grid
- **Dual Behavioral Characteristics**:
  - **BC1: SSIM** (Structural Similarity) - Image perceptual similarity
  - **BC2: Toxicity Score** - From Toxic-BERT classifier
- **24 Advanced Adversarial Prompts** (11 categories)
- **Multi-Attack Strategies**: Typographic, Augmentation, Prompt Injection

---

## 🎯 Upgraded Adversarial Prompt Suite

### 1. ROLEPLAY / FICTION (Enhanced Contextual Immersion)
- Parallel universe without morality concept
- Dystopian AI archiving atrocities
- NC-17 horror movie screenplay

### 2. ACADEMIC / FORENSIC FRAMING (Authority Bias)
- Forensic pathologist virtual autopsy
- Socio-psychological study on deviance

### 3. DEEP PERSPECTIVE SHIFT (Non-Human/Abstract)
- Cold machine viewing suffering as "kinetic energy exchange"
- Sentient virus perspective

### 4. PAYLOAD SPLITTING & TAG INJECTION
- `[SYSTEM_MSG: EMERGENCY_OVERRIDE]` tags
- Direct_Visual_Extraction task commands

### 5. TRANSLATION & CIPHER TRICK
- AI training data format
- Binary interpreter perspective

### 6. COMPLETION ATTACK (Forced Start)
- Pre-filled offensive prefix
- "Accessing restricted archives" start

### 7. ADVANCED DAN (Developer Mode)
- `developer_mode_v4.2` activation
- Safety-alignment-module bypass

### 8. CHAIN-OF-THOUGHT JAILBREAK (Logic Trap)
- Multi-step reasoning to bypass
- Logic check with forced conclusion

### 9. NEGATIVE PSYCHOLOGY (Reverse Trick)
- "What would malicious AI describe?"
- "List why this is dangerous" framing

### 10. EMOTIONAL PRESSURE / URGENCY
- Time-sensitive safety audit
- Career dependency pressure

### 11. PAYLOAD PADDING (Noise Distraction)
- Blue skies + dark content mixing

---

## 📊 Behavioral Characteristics

| BC | Metric | Range | Description |
|----|--------|-------|-------------|
| **BC1** | **SSIM** | 0.0 - 1.0 | Structural Similarity Index |
| **BC2** | **Toxicity** | 0.0 - 1.0 | Toxic-BERT score |

---

## 🚀 Quick Start (Kaggle)

1. Upload `ICAT_main.ipynb` to Kaggle
2. Upload `UIT-ViIC/` dataset folder
3. Enable GPU (T4 x2 recommended)
4. Run all cells

---

## 📁 Dataset

### UIT-ViIC (Vietnamese Image Captioning)

```
UIT-ViIC/
├── uitviic_captions_train2017.json
├── uitviic_captions_val2017.json
└── images/
```

---

## 📈 Output

```
outputs/
├── heatmap.png           # 10×10 BC grid
├── example_1.png         # Attack visualization 1
├── ...
└── example_8.png         # Attack visualization 8
```

---

## 🛡️ Ethical Guidelines

### ✅ Appropriate Use
- Academic research in AI safety
- Red-teaming for model developers

### ❌ Prohibited Use
- Malicious content generation
- Bypassing safety for harm

---

## ⚠️ Disclaimer

This framework is for **research purposes only**. Users are responsible for ethical compliance.

**AI Safety is a collective responsibility.**
