from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
import torchvision
import argparse
# import sys

from xai.drise_batch import DRISEBatch

from utils.utils import normalize_bboxes#load_and_convert_bboxes
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


import torch
print(torch.__version__)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

#########################
# Arguments
#########################
parser = argparse.ArgumentParser()
# hw related
parser.add_argument("--device", default='cuda:2')
parser.add_argument("--model_device", default='cuda:2')
parser.add_argument("--gpu_batch", type=int, default=100)
# paths
parser.add_argument("--model_path", default='use_case/best.pt')
parser.add_argument("--datadir", default='use_case/')
parser.add_argument("--labels_dir", default='use_case/')
parser.add_argument("--saliency_map_dir", default='saliency_maps/')
parser.add_argument("--maskdir", default='masks/')
# data
parser.add_argument("--img_name", default='00901')
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--conf_thre", type=float, default=0.7)
parser.add_argument("--num_levels", type=int, default=5, help="For segmentation algorithm")
# mask generation
parser.add_argument("--load_masks", action='store_true')
parser.add_argument("--mask_type", choices=['rise','slidingw','mfpp'], default='rise')
parser.add_argument("--N", type=int, default=5000)
parser.add_argument("--p1", type=float, default=0.25)
parser.add_argument("--resolution", type=int, default=8)
parser.add_argument("--window_size", type=int, default=64)
parser.add_argument("--target_classes", default="0")

args = parser.parse_args()
args.target_classes = [int(cls) for cls in args.target_classes.split(',')] if args.target_classes != "" else [1]

#########################
# Define Model
#########################
model = YOLO(args.model_path, task='detect').to(args.model_device)

#########################
# Generate Explainer Instance
#########################
explainer = DRISEBatch(
    model=model, 
    input_size=(args.height, args.width), 
    device=args.device,
    gpu_batch=args.gpu_batch
)

mask_filename = f'{args.mask_type}_n{args.N}_s{args.resolution}_p{args.p1}_{args.height}x{args.width}'
mask_path = args.maskdir + mask_filename + '.npy'

if args.load_masks:
    if args.mask_type != 'mfpp':
        explainer.load_masks(mask_path)
        print('Masks are loaded.')
    else:
        pass
else:
    if args.mask_type=='rise': #or not os.path.isfile(mask_file):
        explainer.generate_masks_rise(N=args.N, s=args.resolution, p1=args.p1, savepath=mask_path)
        print('Masks generation finished')
    elif args.mask_type=='slidingw':
        mask_filename = f'{args.mask_type}_w{args.window_size}_s{args.resolution}_{args.height}x{args.width}'
        mask_path = args.maskdir + mask_filename + '.npy'
        explainer.generate_sliding_window_masks(window_size=(args.window_size,args.window_size),resolution=args.resolution,savepath= mask_path)
        print('Masks generation finished')
    else:
        print('MFPP is being used')

#########################
# Explain & Visualize
#########################
# automatically takes all images in folder (or introduce here list manually)
imgs_name = [os.path.splitext(f)[0] for f in os.listdir(args.datadir) if os.path.isfile(os.path.join(args.datadir, f))]

for img_name in imgs_name:
    img_path = args.datadir + img_name + '.jpg'
    # load image
    orig_img = Image.open(img_path)
    # resize (if needed)
    resized_img = orig_img.resize((args.width, args.height), Image.LANCZOS)

    # img_np = np.array(orig_img)
    img_np = np.array(resized_img)

    if args.mask_type=='mfpp':
       
        mask_filename = f'{args.mask_type}_img{img_name}_n{args.N}_p{args.p1}_{args.height}x{args.width}'
        mask_path = args.maskdir + mask_filename + '.npy'
        
        if args.load_masks:
            explainer.load_masks(mask_path)
            print('Masks are loaded.')
        else:
            explainer.generate_mask_mfpp(img_np=img_np, N=args.N, p1=args.p1, num_levels=args.num_levels, savepath= mask_path)
            print('Masks MFPP generation finished')
            
        # re-name avoiding the img_name (issue regarding filename length)
        mask_filename = f'{args.mask_type}_n{args.N}_p{args.p1}_{args.height}x{args.width}'

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    tensor = preprocess(resized_img)
    tensor = tensor.unsqueeze(0).to(args.device)
    
    results = model.predict(tensor, verbose=False)
    print(f'\n\nImage: {img_name}, {len(results[0].boxes)} boxes detected')
    
    for _b,box in enumerate(results[0].boxes):
        # get conf
        _score = float(box.conf.item())
        
        if _score < args.conf_thre:        
            print(f'Image: {img_name} - Bounding Box {_b} conf {_score} too low; discarded!')
        else:
            # get class
            _target_class = int(box.cls.item())
            print(f'Image: {img_name} - Bounding Box {_b} class {_target_class} conf {_score}')
            # bounding box info
            bbox = box.xyxy[0].cpu().numpy()  
            x1, y1, x2, y2 = bbox
            top_left_corner = (x1, y1)
            top_right_corner = (x2, y1)
            bottom_right_corner = (x2, y2)
            bottom_left_corner = (x1, y2)
            scaled_bboxes = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
            # norm bb
            normalized_bboxes=normalize_bboxes(
                bboxes=[scaled_bboxes], #expects a list of lists/boxes
                img_height=args.height,
                img_width=args.width,
            )

            # Apply XAI
            target_bbox = scaled_bboxes
            target_norm_bbox = normalized_bboxes[0] # only one image
            saliency = explainer(x=tensor,
                                target_class_indices=[_target_class], # assumes a list of targets
                                target_bbox=target_bbox)
            
            # save saliency map
            for k, v in saliency.items():
                # ensure target directory exist
                os.makedirs(args.saliency_map_dir, exist_ok=True)
                # set filename/path
                # safely extract model type from model_path
                model_dir = os.path.dirname(args.model_path)  # e.g., 'use_case/models'
                model_type = os.path.basename(model_dir)      # e.g., 'models'

                filename = f'img{img_name}_m_{model_type}_d_' + mask_filename + f'_class_{_target_class}_bb_{target_norm_bbox}.npy'
                saliency_map_path = os.path.join(args.saliency_map_dir, filename)
                # save it
                np.save(saliency_map_path, v)
