from setuptools import setup, find_packages

setup(
    name='createOnnx',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'createOnnx': ['data/*.jpg',
                       'data/gt/*.png',
                       'data/lq/*.png',
                       'data/merged/*.jpg',
                       'data/trimap/*.png']
    },
    install_requires=['onnx',
                      'onnx-simplifier',
                      'torch==1.6.0',
                      'torchvision==0.7.0',
                      'mmcls',
                      'mmdet',
                      'mmedit',
                      'mmsegmentation'],

    entry_points={
        'console_scripts': ['createOnnx=createOnnx.createOnnx:main']
    }
)