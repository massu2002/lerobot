#!/usr/bin/env bash
# ===============================================
# SmolVLA Appendix A.1 "List of datasets" 抽出スクリプト
# - Linux / macOS 用
# - PyMuPDF を使用して PDF から dataset ID を抽出
# ===============================================

set -e

# --- 引数確認 ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <SmolVLA.pdfのパス>"
    exit 1
fi

PDF_PATH="$1"
OUT_TXT="dataset_ids.txt"

if [ ! -f "$PDF_PATH" ]; then
    echo "❌ 指定されたPDFファイルが見つかりません: $PDF_PATH"
    exit 1
fi

# --- PyMuPDF チェック ---
if ! python3 -c "import fitz" 2>/dev/null; then
    echo "📦 PyMuPDF をインストールします..."
    pip install --quiet PyMuPDF
fi

TMP_TXT="$(mktemp)"

echo "📖 PDFをテキスト化しています..."
# 重要: 引数は << の “前” に渡す（python3 - "$PDF_PATH" <<'PYCODE'）
python3 - "$PDF_PATH" <<'PYCODE' > "$TMP_TXT"
import sys, re
try:
    import fitz
except Exception as e:
    print("PYERR: need PyMuPDF (pip install PyMuPDF):", e)
    sys.exit(9)

if len(sys.argv) < 2:
    print("PYERR: missing pdf path")
    sys.exit(2)

pdf = sys.argv[1]
doc = fitz.open(pdf)
text = "".join(p.get_text() for p in doc)

start = text.lower().find("list of datasets")
target = text if start < 0 else text[start:]

ids = sorted(set(re.findall(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", target)))
for i in ids:
    print(i)
PYCODE

echo "🧹 重複を削除して保存しています..."
sort -u "$TMP_TXT" > "$OUT_TXT"
rm -f "$TMP_TXT"

COUNT=$(wc -l < "$OUT_TXT" | tr -d ' ')
echo "✅ 抽出完了: ${COUNT} 件のデータセットIDを検出しました。"
echo "📄 保存先: $(realpath "$OUT_TXT")"
