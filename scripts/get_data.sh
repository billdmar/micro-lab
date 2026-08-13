#!/usr/bin/env bash
# get_data.sh — download the free LOBSTER sample limit-order-book files for the
# five mission tickers (AAPL, AMZN, GOOG, INTC, MSFT) at message depth level 10.
#
# LOBSTER (https://lobsterdata.com) reconstructs NASDAQ limit order books from
# TotalView-ITCH and distributes a standard *free academic sample* for a single
# session (2012-06-21). As of 2026 the lobsterdata.com site is a single-page app
# that gates downloads behind a request/approval workflow, so the historical
# direct-download URLs no longer resolve. We therefore pull the identical free
# sample set from a public, no-signup mirror on the Hugging Face Hub. The data
# is unchanged LOBSTER output; see docs/DESIGN.md for the provenance note and the
# README for LOBSTER attribution. Raw files are NEVER committed or redistributed
# (data/raw/ is gitignored) — only small DERIVED fixtures are committed.
#
# Each ticker delivers a matched pair:
#   <TICKER>_2012-06-21_34200000_57600000_message_10.csv   (order-flow events)
#   <TICKER>_2012-06-21_34200000_57600000_orderbook_10.csv (reference book states)
# The orderbook file is our external correctness oracle for the reconstruction
# differential (). message: time,event_type,order_id,size,price,direction.
# orderbook: (ask_px,ask_sz,bid_px,bid_sz) x 10 levels, prices in 1/10000 dollars.
#
# Deterministic, resumable (curl -C -), and idempotent: re-running only fetches
# missing/incomplete files. Usage: bash scripts/get_data.sh [TICKER ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
BASE_URL="https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main"
SESSION="2012-06-21"
WINDOW="34200000_57600000"   # full session 09:30:00–16:00:00 ET, in ms past midnight
LEVEL="10"
DEFAULT_TICKERS=(AAPL AMZN GOOG INTC MSFT)

TICKERS=("$@")
if [ "${#TICKERS[@]}" -eq 0 ]; then
  TICKERS=("${DEFAULT_TICKERS[@]}")
fi

mkdir -p "${RAW_DIR}"
echo "LOBSTER sample download -> ${RAW_DIR}"
echo "Source: ${BASE_URL} (public mirror of LOBSTER free academic sample, ${SESSION})"
echo "Tickers: ${TICKERS[*]}  |  depth level ${LEVEL}"
echo

fetch() {
  local ticker="$1" kind="$2"
  local fname="${ticker}_${SESSION}_${WINDOW}_${kind}_${LEVEL}.csv"
  local subdir="LOBSTER_SampleFile_${ticker}_${SESSION}_${LEVEL}"
  local url="${BASE_URL}/${subdir}/${fname}"
  local out="${RAW_DIR}/${fname}"
  if [ -s "${out}" ]; then
    echo "  [skip] ${fname} ($(du -h "${out}" | cut -f1))"
    return 0
  fi
  echo "  [get ] ${fname}"
  curl -fSL -C - --retry 3 --retry-delay 2 "${url}" -o "${out}"
  echo "         done ($(du -h "${out}" | cut -f1))"
}

for t in "${TICKERS[@]}"; do
  echo "== ${t} =="
  fetch "${t}" message
  fetch "${t}" orderbook
done

echo
echo "All requested LOBSTER sample files present in ${RAW_DIR}."
ls -lh "${RAW_DIR}" | awk 'NR>1 {print "  "$5"\t"$9}'
