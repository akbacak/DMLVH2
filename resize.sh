for f in "/home/ubuntu/Desktop/myDataset2/Frames/*/*.jpg"
do
     mogrify $f -resize 224x224! $f
done

