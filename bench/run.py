"""CLI entrypoint for the benchmark harness."""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click

from adapters.base import MemoryAdapter
from judge.llm_judge import make_llm_call, score_answer

RESULTS_DIR = Path(__file__).parent / "results"
DATASETS_DIR = Path(__file__).parent / "datasets"

ADAPTER_REGISTRY: dict[str, type] = {}


def _register_adapters() -> None:
    """Lazily import and register available adapters."""
    from adapters.fullcontext import FullContextAdapter
    from adapters.naive_rag import NaiveRAGAdapter

    ADAPTER_REGISTRY["fullcontext"] = FullContextAdapter
    ADAPTER_REGISTRY["naive_rag"] = NaiveRAGAdapter

    try:
        from adapters.mem0_adapter import Mem0Adapter
        ADAPTER_REGISTRY["mem0"] = Mem0Adapter
    except ImportError:
        pass

    try:
        from adapters.zep_adapter import ZepAdapter
        ADAPTER_REGISTRY["zep"] = ZepAdapter
    except ImportError:
        pass

    try:
        from adapters.alaya import AlayaAdapter
        ADAPTER_REGISTRY["alaya"] = AlayaAdapter
    except (ImportError, FileNotFoundError):
        pass


def _print_table(results: list[dict]) -> None:
    """Print results as a formatted table matching the paper."""
    print("\n" + "=" * 60)
    print(f"{'System':<20} {'Accuracy (%)':<15} {'Correct':<10} {'Total':<10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{r['system']:<20} {r['accuracy'] * 100:>10.2f}%    "
              f"{r['correct']:<10} {r['total']:<10}")
    print("=" * 60)


@click.command()
@click.argument("benchmark", type=click.Choice(["locomo", "longmemeval"]))
@click.option("--systems", "-s", default="fullcontext,naive_rag",
              help="Comma-separated adapter names")
@click.option("--limit", "-n", type=int, default=None,
              help="Limit number of questions")
@click.option("--dry-run", is_flag=True, help="Estimate cost without running")
@click.option("--dataset", type=click.Path(exists=True), default=None,
              help="Override dataset path")
def cli(benchmark: str, systems: str, limit: int | None, dry_run: bool, dataset: str | None):
    """Run memory system benchmarks.

    Examples:
        python run.py locomo --systems fullcontext,naive_rag --limit 10
        python run.py longmemeval --systems alaya,mem0 --dry-run
    """
    _register_adapters()

    system_names = [s.strip() for s in systems.split(",")]
    adapters: list[MemoryAdapter] = []
    for name in system_names:
        if name not in ADAPTER_REGISTRY:
            available = ", ".join(ADAPTER_REGISTRY.keys())
            click.echo(f"Unknown adapter: {name}. Available: {available}", err=True)
            sys.exit(1)
        adapters.append(ADAPTER_REGISTRY[name]())

    # Resolve dataset path
    if dataset:
        dataset_path = Path(dataset)
    elif benchmark == "locomo":
        dataset_path = DATASETS_DIR / "locomo10.json"
    else:
        dataset_path = DATASETS_DIR / "longmemeval_s.json"

    if not dataset_path.exists():
        click.echo(f"Dataset not found: {dataset_path}. Run: python datasets/download.py", err=True)
        sys.exit(1)

    llm_call_fn = make_llm_call()

    def judge_fn(question: str, gold: str, prediction: str) -> float:
        return score_answer(question, gold, prediction)

    all_results = []

    for adapter in adapters:
        click.echo(f"\nRunning {benchmark} on {adapter.name} ...")

        if benchmark == "locomo":
            from runners.locomo import run_locomo
            result = run_locomo(adapter, dataset_path, judge_fn, llm_call_fn,
                                limit=limit, dry_run=dry_run)
        else:
            from runners.longmemeval import run_longmemeval
            result = run_longmemeval(adapter, dataset_path, judge_fn, llm_call_fn,
                                     limit=limit, dry_run=dry_run)

        summary = {
            "system": result.system,
            "accuracy": result.accuracy,
            "correct": result.correct,
            "total": result.total_questions,
            "elapsed_seconds": result.elapsed_seconds,
        }
        all_results.append(summary)

        if not dry_run:
            # Save full results
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = RESULTS_DIR / f"{benchmark}_{adapter.name}_{ts}.json"
            with open(out_path, "w") as f:
                json.dump(asdict(result), f, indent=2)
            click.echo(f"Results saved: {out_path}")

    _print_table(all_results)


if __name__ == "__main__":
    cli()
