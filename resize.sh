for f in "CPSM_images/*/*.jpg"
do
     mogrify $f -resize 224x224! $f
done

