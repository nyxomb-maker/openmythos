🧠 OpenMythos Clone
Hybrid Iterative MoE Language Model (PyTorch)

🚀 Overview

Antigravity Prototype is a minimal, trainable hybrid language model architecture that combines:

iterative reasoning
sparse Mixture of Experts (MoE)
adaptive computation
hierarchical memory
sub-agent delegation
tool usage

This repository is a research-oriented prototype focused on exploring how these mechanisms interact in a unified system — not a production-ready model.

🧠 Core Idea

Instead of scaling parameters blindly, build models that:
think iteratively, route computation, remember selectively, delegate tasks, and use tools.

🏗️ Architecture
Key Components
16-layer transformer-like stack
Recurrent reasoning loop (per layer)
Adaptive Computation Time (ACT)
Mixture of Experts (MoE)
32 experts per layer
top-8 active per token
Attention over previous reasoning states
Hierarchical memory system
fast memory (recent states)
slow memory (vector retrieval)
compressed storage (autoencoder)
Sub-agent framework
Tool calling system
⚙️ Features
✅ Iterative reasoning (multi-step inference)
✅ Dynamic computation depth (ACT)
✅ Sparse expert routing (MoE 32→8)
✅ Load balancing loss
✅ Memory-augmented processing
✅ Sub-agent invocation
✅ External tool usage
✅ Trainable end-to-end
✅ Runs on single GPU or CPU
📊 Logging & Observability

The model logs:

number of reasoning steps (ACT)
expert usage distribution
expert sparsity patterns
sub-agent calls
tool usage frequency

🧪 Configuration (Default)
Parameter	Value
Layers	16
Dim	128–192
Heads	4
Experts	32/layer
Top-K	8
Steps (ACT)	4–6
Seq Length	64–128
🧠 System Breakdown
🔁 Recurrent Reasoning

Each layer runs multiple internal steps, refining its representation before passing forward.

🧩 Mixture of Experts (MoE)

Sparse routing:

only 8 out of 32 experts activated per token
improves capacity without full compute cost
⏱️ Adaptive Computation Time (ACT)

The model decides:

how many reasoning steps to take
when to halt computation
🧠 Memory System

Fast Memory

recent hidden states

Slow Memory

vector similarity retrieval

Compression

autoencoder reduces storage size
🤖 Sub-Agents

Specialized modules:

reasoning agent
math agent
memory agent

The model learns when to delegate.

🔧 Tool Calling

Example tools:

calculator
string manipulation
memory lookup

The model:

decides to call a tool
selects which tool
generates input
integrates the result
📁 Project Structure
.
├── main.py              # full prototype (model + training)
├── models/             # core modules (optional split)
├── memory/             # memory system
├── agents/             # sub-agents
├── tools/              # tool definitions
├── data/               # toy dataset
├── logs/               # training logs
└── README.md
⚠️ Limitations
Not optimized for performance
MoE routing is simplified
Tool system is basic (no structured parsing)
Memory system is approximate (no full ANN unless extended)
Not suitable for large-scale training
🛠️ Future Work
Efficient MoE routing (CUDA / token batching)
FAISS integration for scalable memory
Better tool interface (structured arguments / parsing)
Multi-agent coordination
Instruction tuning / RL
Scaling experiments
🤝 Contributing

Contributions are welcome — especially in:

performance optimization
better routing strategies
new agents/tools
memory improvements
📜 License

MIT License
This project is an experiment in compositional intelligence:

combining reasoning, memory, modularity, and action into a single small model.
