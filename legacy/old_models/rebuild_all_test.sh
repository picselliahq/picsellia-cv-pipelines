docker build . -f tf2/Dockerfile -t picsellia/training-tf2:test
docker push picsellia/training-tf2:test

docker build . -f keras_classification/Dockerfile -t picsellia/training-keras-classification:test
docker push picsellia/training-keras-classification:test

docker build . -f yolov5_detection/Dockerfile -t picsellia/training-yolov5-detection:test
docker push picsellia/training-yolov5-detection:test

docker build . -f yolov5_segmentation/Dockerfile -t picsellia/training-yolov5-segmentation:test
docker push picsellia/training-yolov5-segmentation:test

docker build . -f unet_instance_segmentation/Dockerfile -t picsellia/training-unet-segmentation:test
docker push picsellia/training-unet-segmentation:test

docker build . -f ViT_classification/Dockerfile -t picsellia/training-vit-classification:test
docker push picsellia/training-vit-classification:test

docker build . -f yolox_detection/Dockerfile -t picsellia/training-yolox-detection:test
docker push picsellia/training-yolox-detection:test
