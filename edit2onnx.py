import mmcv
import warnings
import os
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmedit.datasets.pipelines import Compose
from mmedit.models import build_model

merged_img = os.path.join(os.path.dirname(__file__), './data/merged/GT05.jpg')
trimap_img = os.path.join(os.path.dirname(__file__), './data/merged/GT05.jpg')

lq_img = os.path.join(os.path.dirname(__file__), './data/img.jpg')
gt_img = os.path.join(os.path.dirname(__file__), './data/img.jpg')

def convertEdit2Onnx(config,
                     checkpoint,
                     edit_class='mattors',
                     opset_version=11,
                     show=False,
                     do_simplify=False,
                     output_file='tmp.onnx',
                     save_input=False,
                     save_output=False,
                     verify=False):

    config = mmcv.Config.fromfile(config)
    config.model.pretrained = None
    # ONNX does not support spectral norm
    if hasattr(config.model, 'backone') and hasattr(config.model.backbone.encoder, 'with_spectral_norm'):
        config.model.backbone.encoder.with_spectral_norm = False
        config.model.backbone.decoder.with_spectral_norm = False
    config.test_cfg.metrics = None

    # build the model
    model = build_model(config.model, test_cfg=config.test_cfg)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

    model.cpu().eval()
    model.forward = model.forward_dummy

    # remove alpha from test_pipeline
    keys_to_remove = ['alpha', 'ori_alpha']
    for key in keys_to_remove:
        for pipeline in list(config.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                config.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    config.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(config.test_pipeline)
    # prepare data
    try:
        if edit_class == 'mattors':
            data = dict(merged_path=merged_img, trimap_path=trimap_img)
            input = test_pipeline(data)

            merged = input['merged'].unsqueeze(0)
            trimap = input['trimap'].unsqueeze(0)
            input = torch.cat((merged, trimap), 1)
        elif edit_class == 'restorers':
            data = dict(lq_path=lq_img, gt_path=gt_img)
            input = test_pipeline(data)
            input = input['lq'].unsqueeze(0)
        else:
            raise ValueError('edit_class {} is not support, please chose mattors or restorers'.format(edit_class))
    except ValueError:
        raise

    # pytorch has some bug in pytorch1.3, we have to fix it
    # by replacing these existing op
    register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            output_file,
            input_names=['input'],
            export_params=True,
            keep_initializers_as_inputs=True,
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
                                         custom_lib=ort_custom_op_path)
            onnx.save(onnx_opt_model, output_file)
    print(f'Successfully exported ONNX model: {output_file}')

    if save_input:
        input.detach().numpy().tofile('input.bin')

    if not save_output and not verify:
        return

    # check by onnx
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    # get onnx output
    sess = rt.InferenceSession(output_file)
    onnx_result = sess.run(None, {
        'input': input.detach().numpy(),
    })
    # only concern pred_alpha value
    if isinstance(onnx_result, (tuple, list)):
        onnx_result = onnx_result[0]

    if save_output:
        np.array(onnx_result).tofile('output.bin')

    if verify:
        # get pytorch output, only concern pred_alpha
        pytorch_result = model(input)
        if isinstance(pytorch_result, (tuple, list)):
            pytorch_result = pytorch_result[0]
        pytorch_result = pytorch_result.detach().numpy()
        # check the numerical value
        try:
            assert np.allclose(pytorch_result, onnx_result)
            print('The numerical values are same between Pytorch and ONNX')
        except AssertionError:
            print('The outputs are different between Pytorch and ONNX')