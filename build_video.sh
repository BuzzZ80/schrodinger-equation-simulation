ffmpeg -framerate 24 -i output/img%04d.bmp -c:v libx264 -crf 1 -pix_fmt yuv420p output.mp4
