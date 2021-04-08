import argparse
import os
from detection2onnx import convertDetection2Onnx
from cls2onnx import convertCls2Onnx
from edit2onnx import convertEdit2Onnx
from seg2onnx import convertSeg2Onnx
def parse_args():
    parser = argparse.ArgumentParser(description='convert mmlab model to ONNX')
    parser.add_argument('--class-name', type=str, help='which model class you want to convert, such as: detection/classification/editing/segmentation')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--checkpoint', help='checkpoint file path')
    parser.add_argument('--dynamic-shape', action='store_true', help='whether to export onnx with dynamic shape')
    parser.add_argument('--onnx-name', type=str, help='onnx model file name')
    parser.add_argument('--simplify', action='store_true', help='whether to simplify onnx model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_img = os.path.join(os.path.dirname(__file__), './data/img.jpg')
    input_shape = (1,3,400,800)
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    merged_img = os.path.join(os.path.dirname(__file__), './data/merged/GT05.jpg')
    trimap_img = os.path.join(os.path.dirname(__file__), './data/merged/GT05.jpg')
    try:
        if args.class_name is None:
            raise RuntimeError()
    except RuntimeError:
        print('please use --class-name to chose which model class to convert, such as: --class-name mmdetection, --class-name mmclassification, --class-name mmediting, --class-name mmsegmentation')
        exit(0)
    try:
        if args.config is None:
            raise RuntimeError()
    except RuntimeError:
        print('please use --config xxx.pth to chose config file')
        exit(0)
    try:
        if args.checkpoint is None:
            raise RuntimeError()
    except RuntimeError:
        print('please use --checkpoint xxx.checkpoint to chose checkpoint file')
        exit(0)

    out_file = args.onnx_name
    if out_file is None:
        out_file = 'tmp.onnx'
        print('onnx model name is default: tmp.onnx')

    try:
        if args.class_name == 'detection':
            normalize_cfg = {'mean': mean, 'std': std}
            convertDetection2Onnx(args.config,
                                  args.checkpoint,
                                  input_img,
                                  input_shape,
                                  opset_version=11,
                                  output_file=out_file,
                                  verify=False,
                                  normalize_cfg=normalize_cfg,
                                  dynamic_export=args.dynamic_shape,
                                  do_simplify=args.simplify)
        elif args.class_name == 'classification':
            convertCls2Onnx(args.config,
                            args.checkpoint,
                            input_shape,
                            output_file=out_file)
        elif args.class_name == 'editing':
            convertEdit2Onnx(args.config,
                             args.checkpoint,
                             merged_img,
                             trimap_img,
                             output_file=out_file)

        elif args.class_name == 'segmentation':
            convertSeg2Onnx(args.config,
                            args.checkpoint,
                            input_shape,
                            output_file=out_file)
        else:
            raise RuntimeError()
    except RuntimeError:
        print('class name {0} is not support, please use: {1}, {2}, {3}, {4}'.format(args.class_name,
                                                                                     'detection',
                                                                                     'classification',
                                                                                     'editing',
                                                                                     'segmentation'))
