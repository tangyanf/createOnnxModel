import argparse
import os
from detection2onnx import convertDetection2Onnx
from cls2onnx import convertCls2Onnx
from edit2onnx import convertEdit2Onnx
from seg2onnx import convertSeg2Onnx
def parse_args():
    parser = argparse.ArgumentParser(description='convert mmlab model to ONNX')
    parser.add_argument('--detection', action='store_true', help='convert detection model')
    parser.add_argument('--classification', action='store_true', help='convert classification model')
    parser.add_argument('--editing', action='store_true', help='convert editing model')
    parser.add_argument('--segmentation', action='store_true', help='convert segmentation model')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--checkpoint', help='checkpoint file path')
    parser.add_argument('--dynamic-shape', action='store_true', help='whether to export onnx with dynamic shape')
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
    if args.detection:
        normalize_cfg = {'mean': mean, 'std': std}
        convertDetection2Onnx(args.config,
                              args.checkpoint,
                              input_img,
                              input_shape,
                              opset_version=11,
                              output_file='tmp.onnx',
                              verify=False,
                              normalize_cfg=normalize_cfg,
                              dynamic_export=args.dynamic_shape,
                              do_simplify=args.simplify)
    if args.classification:
        convertCls2Onnx(args.config,
                        args.checkpoint,
                        input_shape)
    if args.editing:
        convertEdit2Onnx(args.config,
                         args.checkpoint,
                         merged_img,
                         trimap_img)

    if args.segmentation:
        convertSeg2Onnx(args.config,
                        args.checkpoint,
                        input_shape)