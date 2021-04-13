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
    install_requires=['mmcls',
                      'mmdet',
                      'mmedit',
                      'mmsegmentation',
                      'torchvision==0.7.0',
                      'torch==1.6.0',
                      'onnx-simplifier',
                      'onnx',
                      'numpy',
                      'Cython'],

    entry_points={
        'console_scripts': ['createOnnx=createOnnx.createOnnx:main']
    }
)