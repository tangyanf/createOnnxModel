# createOnnxModel
a tool to convert mmlab model to onnx

## requirement
* python == 3.8
* mmcv-full

    to install mmcv-full with custom ops, we need to download onnxruntime library, And add it into environment
    1. download onnxruntime-1.5.1
    ```shell
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.5.1/onnxruntime-linux-x64-1.5.1.tgz
    
    tar -zxvf onnxruntime-linux-x64-1.5.1.tgz
    ```
    2. add environment variable
    ```shell
    cd onnxruntime-linux-x64-1.5.1
    export ONNXRUNTIME_DIR=$(pwd)
    export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
    ```
    3. build on linux
    ```shell
    git clone https://github.com.cnpmjs.org/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 MMCV_WITH_ORT=1 pip install -e .
    ```

## build createOnnx
```shell
git clone https://github.com.cnpmjs.org/tangyanf/createOnnxModel.git
cd createOnnxModel
python setup.py install
```

## usage
```shell
createOnnx --class-name detection \
           --config xxx.pth \
           --checkpoint xxx.point \
           --simplify \
           --save-input \
           --save-output \
           --veirfy
```
### Parameter Description
* --class-name:&nbsp;&nbsp;which class to convert(detection/classification/segmentation/editin)
* --config:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;config file
* --checkpoint:&nbsp;&nbsp;  checkpoint file
* --simplify:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;whether to simplify onnx model
* --save-input:&nbsp;&nbsp;&nbsp;  whether to save model's input
* --save-output:&nbsp; whether to save onnxruntime's output
* --verify&nbsp;:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;whether compare the outputs between Pytorch and ONNX
