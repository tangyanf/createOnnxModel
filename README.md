# createOnnxModel
a tool to convert mmlab model to onnx

## requirement
* python >= 3.5
* pytorch == 1.6
  ```shell
  pip install torch==1.6.0 torchvision==0.7.0
  ```
* onnxruntime==1.5.1
  ```shell
  pip install onnxruntime==1.5.1
  ```
* onnx
  ```shell
  pip install onnx
  ```
* onnx-simplifier
  ```shell
  pip install onnx-simplifier
  ```
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
* mmdetection
  ```shell
  git clone https://github.com.cnpmjs.org/open-mmlab/mmdetection.git
  cd mmdetection
  python setup.py install
  ```

* mmclassification
  ```shell
  git clone https://github.com.cnpmjs.org/open-mmlab/mmclassification.git
  cd mmclassification
  python setup.py install
  ```
* mmediting
  ```shell
  git clone https://github.com.cnpmjs.org/open-mmlab/mmediting.git
  cd mmediting
  python setup.py install
  ```
* mmsegmentation
  ```shell
  git clone https://github.com.cnpmjs.org/open-mmlab/mmsegmentation.git
  cd mmsegmentation
  python setup.py install
  ```


## usage
```shell
python createOnnx.py --
```
