from setuptools import setup, find_packages

setup(
    name='createOnnx',
    version='0.0.1',
    install_requires = ['onnx',
                        'onnx-simplifier',
                        'torch==1.6.0',
                        'torchvision==0.7.0',
                        'mmcls',
                        'mmdet',
                        'mmedit',
                        'mmsegmentation'],

    entry_points={
        'console_scripts':['createOnnx=createOnnx.createOnnx:main',]
    }
)