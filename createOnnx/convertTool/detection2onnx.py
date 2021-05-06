import os.path as osp
import warnings

import numpy as np
import onnx
import onnxruntime as rt
import torch

from mmdet.core.export import (build_model_from_cfg, generate_inputs_and_wrap_model,
                        preprocess_example_input)


def convertDetection2Onnx(config_path,
                          checkpoint_path,
                          input_img,
                          input_shape,
                          opset_version=11,
                          show=False,
                          output_file='tmp.onnx',
                          verify=False,
                          save_input=False,
                          save_output=False,
                          normalize_cfg=None,
                          do_simplify=False,
                          dynamic_export=None):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }

    # prepare original model and meta for verifying the onnx model
    orig_model = build_model_from_cfg(config_path, checkpoint_path)
    one_img, one_meta = preprocess_example_input(input_config)
    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config)
    output_names = ['dets', 'labels']
    if model.with_mask:
        output_names.append('masks')
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        if model.with_mask:
            dynamic_axes['masks'] = {0: 'batch', 1: 'num_dets'}

    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        input_names=['input'],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes)

    model.forward = orig_model.forward

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
                                     input_shapes=dict(input=[1,3,400,800]),
                                     dynamic_input_shape=True,
                                     custom_lib=ort_custom_op_path)
        onnx.save(onnx_opt_model, output_file)

    print(f'Successfully exported ONNX model: {output_file}')

    if save_input:
        one_img.detach().numpy().tofile('input.bin')

    if not save_output and not verify:
        return

    from mmdet.core import get_classes, bbox2result

    ort_custom_op_path = ''
    try:
        from mmcv.ops import get_onnxruntime_op_path
        ort_custom_op_path = get_onnxruntime_op_path()
    except (ImportError, ModuleNotFoundError):
        warnings.warn('If input model has custom op from mmcv, \
            you may have to build mmcv with ONNXRuntime from source.')
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 1)
    session_options = rt.SessionOptions()
    # register custom op for onnxruntime
    if osp.exists(ort_custom_op_path):
        session_options.register_custom_ops_library(ort_custom_op_path)
    sess = rt.InferenceSession(output_file, session_options)
    onnx_outputs = sess.run(None,
                            {net_feed_input[0]: one_img.detach().numpy()})
    output_names = [_.name for _ in sess.get_outputs()]
    output_shapes = [_.shape for _ in onnx_outputs]
    print(f'ONNX Runtime output names: {output_names}, \
        output shapes: {output_shapes}')
    onnx_outputs = [_.squeeze(0) for _ in onnx_outputs]
    ort_dets, ort_labels = onnx_outputs[:2]
    num_classes = len(get_classes('coco'))
    onnx_results = bbox2result(ort_dets, ort_labels, num_classes)
    if model.with_mask:
        segm_results = onnx_outputs[2]
        cls_segms = [[] for _ in range(num_classes)]
        for i in range(ort_dets.shape[0]):
            cls_segms[ort_labels[i]].append(segm_results[i])
        onnx_results = (onnx_results, cls_segms)

    if save_output:
        ort_dets.tofile('det.bin')
        ort_labels.tofile('labels.bin')
        if model.with_mask:
            segm_results = onnx_outputs[2]
            segm_results.tofile('mask.bin')

    if verify:
        # get pytorch output
        pytorch_results = model(tensor_data, [[one_meta]], return_loss=False)
        pytorch_results = pytorch_results[0]
        # compare a part of result
        if model.with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        try:
            for onnx_res, pytorch_res in compare_pairs:
                for o_res, p_res in zip(onnx_res, pytorch_res):
                        np.testing.assert_allclose(
                            o_res,
                            p_res,
                            rtol=1e-03,
                            atol=1e-05,
                        )
            print('The numerical values are the same between Pytorch and ONNX')
        except AssertionError:
            print('the numerical values are not same between pytorch and ONNX')
