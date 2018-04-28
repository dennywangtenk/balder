#!/bin/bash

#workaround for Protobuf version issue
Protobuf_URL=https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip
Protobuf_file=protoc-3.5.1-linux-x86_64.zip

VER=0.4
echo "#############################################################"
echo "# Install and Configure Tensorflow Object Detection API on Ubuntu"
echo "#"
echo "#"
echo "# Author:Denny"
echo "#"
echo "# Version:$VER"
echo "#############################################################"
echo ""

clear
  echo "Please enter your work directory name:"
    read -p "directory name [default:my_tensforflow]:" work_dir
    if [ "$work_dir" = "" ]; then
        work_dir="my_tensforflow"
    fi
  echo "Your work directory: $work_dir"
  
echo "Installing PIP, Python, GIT, CURL, UNZIP..."
sudo apt-get -y install python-pip python-dev git curl unzip

echo "Installing Tensorflow..."
pip install tensorflow

echo "Installing Dependencies..."
sudo pip install Cython
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib

mkdir $work_dir
full_work_dir=`pwd`/$work_dir

echo "Download Object Detection API..."
cd $full_work_dir
git clone https://github.com/tensorflow/models.git

echo "Installing COCO API..."
cd $full_work_dir
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools $full_work_dir/models/research/
    
echo "Installing Protobuf..."
cd $full_work_dir
curl -OL $Protobuf_URL
unzip $Protobuf_file -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/
sudo chown $USER /usr/local/bin/protoc
sudo chown -R $USER /usr/local/include/google
cd $full_work_dir/models/research/
/usr/local/bin/protoc object_detection/protos/*.proto --python_out=.

echo "Configuring PYTHONPATH..."
cd $full_work_dir/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo "export PYTHONPATH=$PYTHONPATH:$full_work_dir:$full_work_dir/slim" >> ~/.bashrc


echo "Validating..."
cd $full_work_dir/models/research/
python object_detection/builders/model_builder_test.py
