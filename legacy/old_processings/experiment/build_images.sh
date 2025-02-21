docker build -f Dockerfile.tensorflow . -t picsellpn/evaluate-tf:capture
docker push picsellpn/evaluate-tf:capture

docker build -f Dockerfile.yolov8 . -t picsellpn/evaluate-yolov8:capture
docker push picsellpn/evaluate-yolov8:capture
