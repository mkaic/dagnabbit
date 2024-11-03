ffmpeg \
-framerate 30 \
-i "dagnabbit/outputs/timelapse/%6d.jpg" \
-vcodec libx264 \
-crf 18 \
-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=-1:720:flags=neighbor" \
"dagnabbit/outputs/timelapse.mp4" -y