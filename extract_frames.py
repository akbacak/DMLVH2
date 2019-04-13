# https://askubuntu.com/questions/1019356/how-can-l-use-ffmpeg-to-extract-frames-with-a-certain-fps-ans-scaling

# https://www.bugcodemaster.com/article/extract-images-frame-frame-video-file-using-ffmpeg

# ffmpeg -i KSBY_2009_09_02.avi -vf fps=1 2%04d.jpg -hide_banner
# for f in "KSBY_2009_06_03.avi/*.jpg"
# do
#     convert $f -resize 50% $f.resized.jpg
# done



import os
import math
import cv2
import re

listing = os.listdir("/home/ubuntu/Desktop/Thesis_Follow_Up_3/Datasets/CPSM/CPSM_videos/")

count = 1
for file in listing:
    video = cv2.VideoCapture("/home/ubuntu/Desktop/Thesis_Follow_Up_3/Datasets/CPSM/CPSM_videos/" + file)
    print(video.isOpened())
    framerate = video.get(5)
    os.makedirs("/home/ubuntu/Desktop/Thesis_Follow_Up_3/Datasets/CPSM/CPSM_images/" + file )
    while (video.isOpened()):
        frameId = video.get(1)
        success,image = video.read()
        #if( image != None ):
        #    image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
        if (success != True):
            break
        if (frameId % math.floor(framerate) == 0):
            filename = "/home/ubuntu/Desktop/Thesis_Follow_Up_3/Datasets/CPSM/CPSM_images/" + file +"/" + file + "_" + str(int(frameId / math.floor(framerate))+1) + ".jpg"
            print(filename)
            cv2.imwrite(filename,image)
    video.release()
    print('done')
    count+=1

