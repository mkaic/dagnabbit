ffmpeg \
-framerate 30 \
-i "outputs/timelapse/%6d.png" \
-vcodec libx264 \
-crf 18 \
-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=-1:720:flags=neighbor" \
"outputs/timelapse.mp4" -y