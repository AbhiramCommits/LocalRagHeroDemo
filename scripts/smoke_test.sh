#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fresh-clone smoke test.
#
# Copies the *tracked* files into a throwaway directory, creates a virtualenv,
# installs the package, builds the corpus and the index, asks one question with
# the offline extractive backend and asserts that a citation is printed.
#
# Installs only the core dependencies (no torch/sentence-transformers) so the
# embedder exercises its deterministic hashing fallback — this keeps the smoke
# test fast and network-light while still proving the end-to-end path.
#
#   PYTHON=/path/to/python3.11 bash scripts/smoke_test.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/localrag-smoke.XXXXXX")"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "smoke: copying tracked files to $WORK"
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ( cd "$ROOT" && git ls-files -z ) | tar --null -T - -cf - | tar -xf - -C "$WORK"
else
    cp -R "$ROOT/." "$WORK/"
    rm -rf "$WORK/.git" "$WORK/artifacts" "$WORK/.cache"
fi

cd "$WORK"
echo "smoke: creating venv"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate

echo "smoke: installing package (core deps)"
python -m pip install -q --upgrade pip
python -m pip install -q -e .

echo "smoke: building corpus"
localrag corpus build >/dev/null

echo "smoke: building index (hashing fallback expected without torch)"
localrag index build

echo "smoke: asking one question"
OUT="$(localrag ask "What does fault code E-318 indicate on the PRS-02 press?" --backend extractive)"
echo "----- answer -----"
echo "$OUT"
echo "------------------"

if echo "$OUT" | grep -qE '(MANUAL-|SPEC-|QC-WI-|INC-)'; then
    echo "SMOKE PASS: citation present in answer"
else
    echo "SMOKE FAIL: no citation found in answer" >&2
    exit 1
fi
