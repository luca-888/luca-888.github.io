#!/bin/bash
# ./speeds.sh 1.1 1.2 1.3  -> build/jev-1080x1920-<倍速>x.mp4：画面按倍速重新渲染（不是抽帧），音轨取 build/mix.wav 整体变速（不变调）。
# 先用 node film.mjs build/jev-1080x1920.mp4 出一次原速成片，得到 build/mix.wav。
cd "$(dirname "$0")"
for s in "$@"; do
  echo "window.SPEED = $s;" > speed.js
  node film.mjs "build/silent-${s}x.mp4" --silent | tail -1
  ffmpeg -y -loglevel error -i "build/silent-${s}x.mp4" -i build/mix.wav -filter:a "atempo=$s" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart "build/jev-1080x1920-${s}x.mp4"
  rm -f "build/silent-${s}x.mp4"
  echo "$s x -> $(ffprobe -v error -show_entries format=duration -of csv=p=0 "build/jev-1080x1920-${s}x.mp4") s"
done
echo "window.SPEED = 1;" > speed.js
