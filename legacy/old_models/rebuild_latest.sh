docker build . -f tf2/Dockerfile -t picsellia/training-tf2:latest
docker push picsellia/training-tf2:latest

docker build . -f keras_classification/Dockerfile -t picsellia/training-keras-classification:latest
docker push picsellia/training-keras-classification:latest

docker build . -f yolov5_detection/Dockerfile -t picsellia/training-yolov5-detection:latest
docker push picsellia/training-yolov5-detection:latest

docker build . -f yolov5_segmentation/Dockerfile -t picsellia/training-yolov5-segmentation:latest
docker push picsellia/training-yolov5-segmentation:latest

docker build . -f unet_instance_segmentation/Dockerfile -t picsellia/training-unet-segmentation:latest
docker push picsellia/training-unet-segmentation:latest

docker build . -f ViT_classification/Dockerfile -t picsellia/training-vit-classification:latest
docker push picsellia/training-vit-classification:latest

docker build . -f yolox_detection/Dockerfile -t picsellia/training-yolox-detection:latest
docker push picsellia/training-yolox-detection:latest
