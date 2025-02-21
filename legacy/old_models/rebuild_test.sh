docker build . -f yolov8-classification/Dockerfile -t picsellia/training-yolov8-classification:test
docker push picsellia/training-yolov8-classification:test

docker build . -f yolov8-detection/Dockerfile -t picsellia/training-yolov8-detection:test
docker push picsellia/training-yolov8-detection:test

docker build . -f yolov8-segmentation/Dockerfile -t picsellia/training-yolov8-segmentation:test
docker push picsellia/training-yolov8-segmentation:test

docker build . -f unet-instance-segmentation/Dockerfile -t picsellia/training-unet-segmentation:test
docker push picsellia/training-unet-segmentation:test
