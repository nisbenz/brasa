#!/usr/bin/env bash
# Download NASA FIRMS country yearly summaries (standard processing, all countries).
# 48 zips, ~9 GB. Resume-safe; 8 parallel downloads; stalled connections retry after 60 s.
# Usage: pipeline/00_download.sh [target_dir]   (default ~/firms)
set -euo pipefail
DIR="${1:-$HOME/firms}"
mkdir -p "$DIR" && cd "$DIR"
B=https://firms.modaps.eosdis.nasa.gov/data/country/zips
{
  for y in $(seq 2000 2025); do echo "$B/modis_${y}_all_countries.zip"; done
  for y in $(seq 2012 2025); do echo "$B/viirs-snpp_${y}_all_countries.zip"; done
  for y in $(seq 2018 2025); do echo "$B/viirs-jpss1_${y}_all_countries.zip"; done
} > urls.txt
xargs -n1 -P8 wget -c -nv --tries=20 --retry-connrefused --waitretry=5 --read-timeout=60 < urls.txt
echo "Downloaded $(ls ./*_all_countries.zip | wc -l) of $(wc -l < urls.txt) files into $DIR"
