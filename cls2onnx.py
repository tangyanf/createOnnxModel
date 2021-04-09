from functools import partial

import mmcv
import onnx
import warnings
import numpy as np
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier

torch.manual_seed(3)


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def convertCls2Onnx(config,
                    checkpoint,
                    input_shape,
                    opset_version=11,
                    dynamic_shape=False,
                    do_simplify=False,
                    save_input=False,
                    save_output=False,
                    show=False,
                    output_file='tmp.onnx',
                    verify=False):
    cfg = mmcv.Config.fromfile(config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    model = build_classifier(cfg.model)

    load_checkpoint(model, checkpoint, map_location='cpu')

    model.cpu().eval()

    num_classes = model.head.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas={}, return_loss=False)
    register_extra_symbolics(opset_version)

    # support dynamic shape export
    if dynamic_shape:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'probs': {
                0: 'batch'
            }
        }
    else:
        dynamic_axes = {}

    with torch.no_grad():
        torch.onnx.export(
            model, (img_list, ),
            output_file,
            input_names=['input'],
            output_names=['probs'],
            export_params=True,
            keep_initializers_as_inputs=True,
            dynamic_axes=dynamic_axes,
            verbose=show,
            opset_version=opset_version)

        # simplify onnx model
        if do_simplify:
            from onnxsim import simplify

            ort_custom_op_path = ''
            try:
                from mmcv.ops import get_onnxruntime_op_path
                ort_custom_op_path = get_onnxruntime_op_path()
            except (ImportError, ModuleNotFoundError):
                warnings.warn('If input model has custom op from mmcv, \
                    you may have to build mmcv with ONNXRuntime from source.')

            onnx_opt_model, _ = simplify(output_file,
                                         check_n=0,
                                         skip_fuse_bn=True,
                                         skip_shape_inference=True,
                                         input_shapes=dict(input=[1, 3, 400, 800]),
                                         dynamic_input_shape=True,
                                         custom_lib=ort_custom_op_path)
            onnx.save(onnx_opt_model, output_file)
        print(f'Successfully exported ONNX model: {output_file}')

    if save_input:
        img_list[0].detach().numpy().tofile('input.bin')

    if not save_output and not verify:
        return

    # check by onnx
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 1)
    sess = rt.InferenceSession(output_file)
    print(img_list[0].detach().numpy().dtype)
    print(img_list[0].detach().numpy().shape)
    onnx_result = sess.run(
        None, {net_feed_input[0]: img_list[0].detach().numpy()})[0]

    if save_output:
        np.array(onnx_result).tofile('output.bin')

    if verify:
        # check the numerical value
        # get pytorch output
        model.forward = origin_forward
        pytorch_result = model(img_list, img_metas={}, return_loss=False)[0]

        try:
            assert np.allclose(pytorch_result, onnx_result)
            print('The outputs are same between Pytorch and ONNX')
        except AssertionError:
            print('The outputs are different between Pytorch and ONNX')