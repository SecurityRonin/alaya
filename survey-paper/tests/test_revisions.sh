#!/usr/bin/env bash
# TDD test harness for peer-review revision items (Items 1-3)
# RED:  run before implementation — all tests should fail
# GREEN: run after implementation — all tests must pass
#
# Usage: bash survey-paper/tests/test_revisions.sh
# Exit code 0 = all pass; non-zero = failures exist

set -euo pipefail
PAPER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAILURES=0

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); }

echo "=== Item 1: Evidence-level markers ==="

# 1.1 – Four evidence-level macros defined in main.tex
for macro in evD evS evV evI; do
    grep -q "\\\\newcommand{\\\\${macro}}" "$PAPER_DIR/main.tex" \
        && pass "$macro macro defined in main.tex" \
        || fail "$macro macro NOT defined in main.tex"
done

# 1.2 – At least one evidence marker appears in §05
grep -q '\\ev[DSVI]' "$PAPER_DIR/sections/05-systems.tex" \
    && pass "Evidence markers present in 05-systems.tex" \
    || fail "No evidence markers found in 05-systems.tex"

# 1.3–1.7 – Each commercial paragraph has at least one marker
for system in "ChatGPT Memory" "Claude Memory" "Gemini Memory" "Grok Memory" "Meta AI Memory"; do
    # Extract from the paragraph heading to the next \paragraph{
    if sed -n "/\\\\paragraph{${system}/,/\\\\paragraph{/p" \
           "$PAPER_DIR/sections/05-systems.tex" 2>/dev/null \
       | grep -q '\\ev[DSVI]'; then
        pass "${system} paragraph has evidence marker"
    else
        fail "${system} paragraph missing evidence marker"
    fi
done

# 1.8 – Evidence-level explanation referenced from §02 methodology
grep -q 'evlevel\|evidence.level\|Evidence.Level\|sec:appendix-evidence' \
    "$PAPER_DIR/sections/02-methodology.tex" \
    && pass "Evidence-level explanation referenced in 02-methodology.tex" \
    || fail "Evidence-level explanation NOT referenced in 02-methodology.tex"

echo ""
echo "=== Item 2: Reproducibility appendix ==="

# 2.1 – Appendix file exists
[ -f "$PAPER_DIR/sections/12-appendix.tex" ] \
    && pass "12-appendix.tex exists" \
    || fail "12-appendix.tex does NOT exist"

# 2.2 – Appendix included in main.tex
grep -q '\\input{sections/12-appendix}' "$PAPER_DIR/main.tex" \
    && pass "Appendix \\input present in main.tex" \
    || fail "Appendix \\input NOT found in main.tex"

# 2.3 – Appendix contains an evidence-level legend
grep -qi 'Evidence.Level\|evidence level' "$PAPER_DIR/sections/12-appendix.tex" 2>/dev/null \
    && pass "Appendix contains evidence-level legend" \
    || fail "Appendix missing evidence-level legend"

# 2.4 – Appendix references the system matrix table
grep -q 'tab:system-matrix' "$PAPER_DIR/sections/12-appendix.tex" 2>/dev/null \
    && pass "Appendix references tab:system-matrix" \
    || fail "Appendix does NOT reference tab:system-matrix"

# 2.5 – All four level labels defined in appendix
for level in '\[D\]' '\[S\]' '\[V\]' '\[I\]'; do
    grep -q "$level" "$PAPER_DIR/sections/12-appendix.tex" 2>/dev/null \
        && pass "Appendix defines level $level" \
        || fail "Appendix missing definition for $level"
done

# 2.6 – Appendix has a \label for cross-referencing
grep -q 'label{sec:appendix' "$PAPER_DIR/sections/12-appendix.tex" 2>/dev/null \
    && pass "Appendix has \\label" \
    || fail "Appendix missing \\label"

echo ""
echo "=== Item 3: PRISMA flow diagram ==="

# 3.1 – TikZ figure present in 02-methodology.tex
grep -q 'begin{tikzpicture}' "$PAPER_DIR/sections/02-methodology.tex" \
    && pass "TikZ figure present in 02-methodology.tex" \
    || fail "No TikZ figure in 02-methodology.tex"

# 3.2 – All four PRISMA stage counts present
for count in '520' '387' '249' '138'; do
    grep -q "$count" "$PAPER_DIR/sections/02-methodology.tex" \
        && pass "PRISMA count $count present" \
        || fail "PRISMA count $count MISSING"
done

# 3.3 – Figure has the correct label
grep -q 'label{fig:prisma}' "$PAPER_DIR/sections/02-methodology.tex" \
    && pass "PRISMA figure has label{fig:prisma}" \
    || fail "PRISMA figure missing label{fig:prisma}"

# 3.4 – Figure has a caption mentioning PRISMA
grep -q 'caption.*[Pp][Rr][Ii][Ss][Mm][Aa]' "$PAPER_DIR/sections/02-methodology.tex" \
    && pass "PRISMA figure has PRISMA caption" \
    || fail "PRISMA figure missing PRISMA caption"

# 3.5 – Exclusion counts present (111 at full-text stage)
grep -q '111' "$PAPER_DIR/sections/02-methodology.tex" \
    && pass "Full-text exclusion count (111) present" \
    || fail "Full-text exclusion count (111) MISSING"

echo ""
echo "=== Summary ==="
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL TESTS PASSED ✓"
    exit 0
else
    echo "$FAILURES TEST(S) FAILED ✗"
    exit 1
fi
