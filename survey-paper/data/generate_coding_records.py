#!/usr/bin/env python3
"""Parse survey-paper/sections/system-matrix.tex → coding-records.{csv,json}.

The matrix encodes eight binary capability features plus RAG tier for 138 systems.
Full four-axis taxonomy coding (including working/procedural memory subtypes,
lifecycle formation/transformation sub-operations, and preference N/E/T/M codes)
is documented in the paper text; only the matrix columns are emitted here.

Usage (run from survey-paper/data/):
    python generate_coding_records.py
"""

import re
import csv
import json
from pathlib import Path

COLUMNS = [
    "system", "category", "rag_tier",
    "episodic", "semantic", "graph", "fusion",
    "consolidation", "forgetting", "contradiction", "preference",
]

RAG_LABELS = {"N": "Naive", "A": "Advanced", "M": "Modular", "F": "File-Based"}

CATEGORY_RE = re.compile(
    r"\\multicolumn\{10\}\{@\{\}l\}\{\\textit\{([^}]+)\}\}"
)
# Data rows: SystemName & X & ... (9 columns after system name)
ROW_RE = re.compile(
    r"^([A-Za-z0-9][^&\n]+?)&\s*([NAMF])\s*"  # system & rag
    r"&\s*(\$\\bullet\$|)\s*"   # episodic
    r"&\s*(\$\\bullet\$|)\s*"   # semantic
    r"&\s*(\$\\bullet\$|)\s*"   # graph
    r"&\s*(\$\\bullet\$|)\s*"   # fusion
    r"&\s*(\$\\bullet\$|)\s*"   # consolidation
    r"&\s*(\$\\bullet\$|)\s*"   # forgetting
    r"&\s*(\$\\bullet\$|)\s*"   # contradiction
    r"&\s*(\$\\bullet\$|)\s*"   # preference
    r"\\\\"
)


def clean_name(raw: str) -> str:
    # Strip LaTeX commands: \\ \ ~ ^ { }
    s = raw.strip()
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)  # \command
    s = re.sub(r"[\\~^{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_matrix(tex_path: Path) -> list[dict]:
    records = []
    current_category = "Unknown"
    text = tex_path.read_text(encoding="utf-8")

    for line in text.splitlines():
        cat_m = CATEGORY_RE.search(line)
        if cat_m:
            current_category = cat_m.group(1).strip()
            continue

        row_m = ROW_RE.match(line.strip())
        if row_m:
            system_raw, rag = row_m.group(1), row_m.group(2)
            bits = [1 if row_m.group(i) == "$\\bullet$" else 0 for i in range(3, 11)]
            records.append({
                "system": clean_name(system_raw),
                "category": current_category,
                "rag_tier": RAG_LABELS.get(rag, rag),
                "episodic": bits[0],
                "semantic": bits[1],
                "graph": bits[2],
                "fusion": bits[3],
                "consolidation": bits[4],
                "forgetting": bits[5],
                "contradiction": bits[6],
                "preference": bits[7],
            })

    return records


def main() -> None:
    data_dir = Path(__file__).parent
    tex_path = data_dir.parent / "sections" / "system-matrix.tex"

    records = parse_matrix(tex_path)
    print(f"Parsed {len(records)} system records.")

    # CSV
    csv_path = data_dir / "coding-records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {csv_path}")

    # JSON
    json_path = data_dir / "coding-records.json"
    payload = {
        "schema_version": "1.0",
        "description": (
            "Per-system capability coding for 138 LLM agent memory systems. "
            "Columns match system-matrix.tex: RAG tier (Naive/Advanced/Modular/File-Based) "
            "and eight binary capability features. "
            "Full four-axis taxonomy coding (preference N/E/T/M subtypes, "
            "lifecycle sub-operations) is in the paper text."
        ),
        "paper": "A Taxonomy of Memory Architectures for LLM-Based Agents",
        "survey_cutoff": "April 2026",
        "columns": COLUMNS,
        "systems": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
