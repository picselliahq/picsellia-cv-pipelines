docker build . -f yolov8-classification/Dockerfile -t picsellia/training-yolov8-classification:latest
docker push picsellia/training-yolov8-classification:latest

docker build . -f yolov8-detection/Dockerfile -t picsellia/training-yolov8-detection:latest
docker push picsellia/training-yolov8-detection:latest

docker build . -f yolov8-segmentation/Dockerfile -t picsellia/training-yolov8-segmentation:latest
docker push picsellia/training-yolov8-segmentation:latest

docker build . -f unet-instance-segmentation/Dockerfile -t picsellia/training-unet-segmentation:latest
docker push picsellia/training-unet-segmentation:latest
