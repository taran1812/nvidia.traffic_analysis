#!/usr/bin/env bash
# Downloads one UA-DETRAC training sequence for pipeline validation.
# Full dataset: http://detrac-db.rit.albany.edu
# Single sequence (~80MB) — no login required.

set -e

OUT_DIR="$(dirname "$0")/data"
mkdir -p "$OUT_DIR"

echo "UA-DETRAC dataset instructions:"
echo ""
echo "Option A — download full training zip (~1.5GB):"
echo "  wget 'http://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip' -O $OUT_DIR/detrac.zip"
echo "  unzip $OUT_DIR/detrac.zip 'Insight-MVT_Annotation_Train/MVI_20011/*' -d $OUT_DIR/"
echo ""
echo "Option B — use any traffic MP4 directly:"
echo "  python3 pipeline/pipeline.py --input /path/to/traffic.mp4 --output pipeline/output/out.mp4"
echo ""
echo "Option C — convert UA-DETRAC image sequence to MP4 (after Option A):"
echo "  ffmpeg -r 25 -i $OUT_DIR/MVI_20011/img%05d.jpg -c:v libx264 $OUT_DIR/MVI_20011.mp4"
echo "  python3 pipeline/pipeline.py --input $OUT_DIR/MVI_20011.mp4 --output pipeline/output/out.mp4 --fps 25"
