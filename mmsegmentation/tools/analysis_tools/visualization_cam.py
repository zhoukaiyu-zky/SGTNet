from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules


class SemanticSegmentationTarget:
    def __init__(self, category, mask, size, device):
        self.category = category
        self.mask = torch.from_numpy(mask).to(device)
        self.size = size
        self.device = device

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = F.interpolate(model_output, size=self.size, mode='bilinear', align_corners=False)
        model_output = torch.squeeze(model_output, dim=0)
        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='prediction.png')
    parser.add_argument('--cam-file', default='vis_cam.png')
    parser.add_argument('--target-layers', default='backbone.layer4[2]')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = Config.fromfile(args.config)
    cfg.test_pipeline = cfg.test_dataloader.dataset.pipeline
    register_all_modules()
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()
    if device.type == 'cpu':
        model = revert_sync_batchnorm(model)
        model.eval()
    model_for_cam = model.module if hasattr(model, 'module') else model

    result = inference_model(model, args.img)
    show_result_pyplot(model, args.img, result, draw_gt=False, show=False if args.out_file is not None else True, out_file=args.out_file)

    prediction_data = result.pred_sem_seg.data
    pre_np_data = prediction_data.cpu().numpy().squeeze(0)

    tl_str = args.target_layers
    target_layers = [eval(f'model_for_cam.{tl_str}')]

    sem_classes = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    category = sem_class_to_idx["car"]
    mask_float = np.float32(pre_np_data == category)

    image = np.array(Image.open(args.img).convert('RGB'))
    height, width = image.shape[0], image.shape[1]
    rgb_img = np.float32(image) / 255.0

    config = Config.fromfile(args.config)
    image_mean = config.data_preprocessor['mean']
    image_std = config.data_preprocessor['std']

    input_tensor = preprocess_image(
        rgb_img,
        mean=[x / 255.0 for x in image_mean],
        std=[x / 255.0 for x in image_std]
    )
    input_tensor = input_tensor.to(device).float()

    targets = [SemanticSegmentationTarget(category, mask_float, (height, width), device)]

    use_cuda = (device.type == 'cuda')
    with torch.cuda.amp.autocast(enabled=False):
        with GradCAM(model=model_for_cam, target_layers=target_layers, use_cuda=use_cuda) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    Image.fromarray(cam_image).save(args.cam_file)


if __name__ == '__main__':
    main()
