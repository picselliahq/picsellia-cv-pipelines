wget https://github.com/protocolbuffers/protobuf/releases/download/v23.4/protoc-23.4-linux-x86_64.zip
unzip protoc-23.4-linux-x86_64.zip -d $HOME/protoc
export PATH=$HOME/protoc/bin:$PATH
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:~/Documents/Picsellia/picsellia_repos/picsellia-training-engine/tf2/experiment/models/research
