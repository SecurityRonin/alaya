# Alaya Benchmark Harness

Runs LoCoMo and LongMemEval benchmarks against multiple memory systems.

## Setup

```bash
cd bench
pip install -e ".[all,dev]"
```

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."        # Required for LLM calls
export JUDGE_MODEL="gpt-4o-mini"      # Judge model (default: gpt-4o-mini)
export LLM_MODEL="gpt-4o-mini"        # Answer generation model
export ZEP_API_KEY="..."              # Optional: for Zep adapter
```

## Usage

```bash
# Download datasets
python datasets/download.py

# Run LoCoMo on all systems
python run.py locomo --systems alaya,mem0,zep,fullcontext,naive_rag

# Run LongMemEval on specific systems
python run.py longmemeval --systems alaya,fullcontext --limit 50

# Dry run (estimate cost)
python run.py locomo --dry-run
```
