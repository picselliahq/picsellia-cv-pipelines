# docker build -f docker/Dockerfile.tensorflow . -t picsellpn/preannotation-tf:capture
# docker push picsellpn/preannotation-tf:capture

# docker build -f docker/Dockerfile.tensorflow . -t picsellpn/preannotation-yolov3:capture
# docker push picsellpn/preannotation-yolov3:capture

docker build -f docker/Dockerfile.yolov8 . -t picsellpn/preannotation-yolov8:capture
docker push picsellpn/preannotation-yolov8:capture

docker build -f docker/Dockerfile.mask . -t picsellpn/albumentation-segmentation:capture
docker push picsellpn/albumentation-segmentation:capture
