# LongMemEval Baseline Comparison

**Paper**: LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory  
**Authors**: Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu  
**Published**: ICLR 2025 (arXiv:2410.10813)  
**Benchmark**: LongMemEval_S (~115k tokens per question)

---

## Commercial Memory-Augmented Systems

| System | LLM Backend | Accuracy | Notes |
|--------|-------------|----------|-------|
| **Offline Reading** | GPT-4o | **91.84%** | Upper bound: full history provided |
| **ChatGPT** | GPT-4o | 57.73% | 37% drop from offline |
| **ChatGPT** | GPT-4o-mini | 71.13% | |
| **Coze** | GPT-4o | 32.99% | 64% drop from offline |
| **Coze** | GPT-3.5-turbo | 24.74% | |

**Key Finding**: Commercial systems show significant performance degradation when required to maintain memory across sessions, with ChatGPT dropping 37% and Coze dropping 64% compared to offline reading.

---

## Long-Context LLMs (No Chain-of-Note)

| Model | Size | Oracle | LongMemEval_S | Drop |
|-------|------|--------|---------------|------|
| **GPT-4o** | - | 87.0% | 60.6% | 30.3% ↓ |
| **Llama 3.1 Instruct** | 70B | 74.4% | 33.4% | 55.1% ↓ |
| **Llama 3.1 Instruct** | 8B | 71.0% | 45.4% | 36.1% ↓ |
| **Phi-3 128k Instruct** | 14B | 70.2% | 38.0% | 45.9% ↓ |
| **Phi-3.5 Mini Instruct** | 4B | 66.0% | 34.2% | 48.1% ↓ |

**Oracle**: Accuracy when only evidence sessions provided (no haystack)

---

## Long-Context LLMs (With Chain-of-Note)

| Model | Size | Oracle | LongMemEval_S | Drop |
|-------|------|--------|---------------|------|
| **GPT-4o** | - | 92.4% | 64.0% | 30.7% ↓ |
| **Llama 3.1 Instruct** | 70B | 84.8% | 28.6% | 66.3% ↓ |
| **Llama 3.1 Instruct** | 8B | 71.0% | 42.0% | 40.8% ↓ |
| **Phi-3 128k Instruct** | 14B | 72.2% | 34.4% | 52.4% ↓ |
| **Phi-3.5 Mini Instruct** | 4B | 65.2% | 32.4% | 50.3% ↓ |

**Chain-of-Note**: Technique that extracts key information before answering, improving reading performance.

---

## RASPUTIN Performance

| System | Accuracy | vs. Best Baseline |
|--------|----------|-------------------|
| **RASPUTIN** | **91.2%** | +0.3% vs Offline Reading (91.84%) |

**Note**: RASPUTIN achieves near-oracle performance, approaching the upper bound of offline reading with full context access.

---

## Key Insights

1. **Commercial Systems Gap**: ChatGPT and Coze, despite being state-of-the-art, show 30-64% accuracy drops when required to maintain memory across sessions.

2. **Long-Context LLM Limitations**: Even advanced long-context models (GPT-4o, Llama 3.1 70B) show 30-66% performance drops on LongMemEval_S compared to oracle settings.

3. **Chain-of-Note Impact**: While Chain-of-Note helps GPT-4o (+3.4%), it surprisingly hurts Llama 3.1 70B (-4.8%), suggesting model-dependent effectiveness.

4. **RASPUTIN Achievement**: At 91.2%, RASPUTIN nearly matches the offline reading upper bound (91.84%), demonstrating effective long-term memory management without access to full context.

---

## Benchmark Details

- **Total Questions**: 500 manually curated
- **Question Types**: 7 types covering:
  - Single-session user/assistant information extraction
  - Multi-session reasoning
  - Knowledge updates
  - Temporal reasoning
  - Abstention (knowing when to say "I don't know")
  
- **History Length**: 
  - LongMemEval_S: ~115k tokens
  - LongMemEval_M: 500 sessions (~1.5M tokens)

- **Evaluation**: LLM-based evaluation (GPT-4o-2024-08-06) with 97%+ agreement with human experts

---

## References

- **Paper**: https://arxiv.org/abs/2410.10813
- **GitHub**: https://github.com/xiaowu0162/LongMemEval
- **ICLR 2025**: Published as conference paper
