#!/usr/bin/env bash
# Download the reference papers from arXiv into this folder (PDFs are git-ignored).
set -euo pipefail
cd "$(dirname "$0")"
curl -fsSL -o KITE_arXiv-2606.22113v2.pdf https://arxiv.org/pdf/2606.22113v2
curl -fsSL -o GET-Zero_arXiv-2407.15002v2.pdf https://arxiv.org/pdf/2407.15002v2
ls -lh ./*.pdf
