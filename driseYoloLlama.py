
import ssl
import os

from dataProcessing import DataProcessing
from driseExplainer import DRISEExplainer
from llamaVisionModel import LLMAVisionModel
from metrics.utils import calculate_iou
from yoloModel import YoloModel
from datetime import datetime
from utils.compute_iou import compute_iou


ssl._create_default_https_context = ssl._create_unverified_context

class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
args = Args(**{
    'img_name': '00901',
    'model_path': 'use_case/models/best.pt',
    'datadir': 'use_case/',
    'annotations_dir': 'use_case/',
    'device': 'cuda:0',
    'input_size': (480, 640),
    'gpu_batch': 16,
    'mask_type': 'rise',
    'maskdir': 'masks/',
    'N': 1000,
    'resolution': 8,
    'p1': 0.5,
    'target_classes': [0],
    'show_plots': False,
    'run_id_tag': '',
    'instruction': """You are the Visual LLM specializing in detailed explanation chaining for Visual-LLMs.
You are provided with an image, the bounding box predicted by YOLO and a saliency map for one bounding box generated with DRISE.
Give a detailed analysis on Color, Shape and Result to explain how the model reached the bounding box.

Do not make up any information that is not present in the image, bounding boxes or saliency map.
Do not repeat the same information in different sections.
Do not explain any of the used models or concepts. Only focus on the given image, bounding boxes and saliency map.

"""
})

# setup
date_time_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if args.run_id_tag != '':
    date_time_tag += f'_{args.run_id_tag}'
output_path = f'output/{date_time_tag}/{args.img_name}/'
os.makedirs(output_path, exist_ok=True)


# ## Data Processing
dataProcessing = DataProcessing(args, output_path)
resized_img, img_np = dataProcessing.import_data()
tensor = dataProcessing.preprocess_image(img_np)
labels = dataProcessing.load_labels()
target_class = dataProcessing.set_target_class()
bboxes = dataProcessing.plot_bboxes(img_np, labels, target_class)


# ## YOLO
yoloModel = YoloModel(args)
results = yoloModel.predict(tensor)
predicted_bboxes = yoloModel.extract_bboxes(results)
image_with_bboxes = yoloModel.plot_bboxes(img_np, predicted_bboxes, output_path)


# ## D-RISE
driseExplainer = DRISEExplainer(args, yoloModel, generate_new=True)
mask_path = driseExplainer.generate_masks()

output_path_saliency = f'{output_path}saliency/'
os.makedirs(output_path_saliency, exist_ok=True)

# todo : think about more sensible way to label instead of just increasing i
# todo : do something if no match found
# todo : llama input change
# todo make run for multiple target classes and muttiple images
i = 0
for bbox in bboxes:
    # find the predicted bbox (in predicted_bboxes) that matches the bbox given in the labels
    # Find the predicted bbox that best matches the ground-truth bbox
    best_iou = 0
    best_pred_bbox = None

    for pred_bbox in predicted_bboxes:
        iou = calculate_iou(bbox, pred_bbox)
        if iou > best_iou:
            best_iou = iou
            best_pred_bbox = pred_bbox

    # Optionally set a threshold to ensure it's a valid match
    if best_iou < 0.3:
        print(f"No good match found for bbox {bbox} (best IoU: {best_iou:.2f})")
        continue

    
    saliency = driseExplainer.apply_saliency(tensor, bbox)
    # plot salicney map with both target bbox and predicted bbox
    

    driseExplainer.plot_saliency(saliency, bbox, best_pred_bbox, img_np, f'{output_path_saliency}drise_saliency_{i}.png', target_class)
    i += 1


# ## LLAMA VL
llamaVisionModel = LLMAVisionModel(args)
model, tokenizer = llamaVisionModel.load_model()
composed = llamaVisionModel.createImageInput(output_path, resized_img, target_class)
inputs = llamaVisionModel.compose_input(tokenizer, composed)
llamaVisionModel.generate_response(inputs, tokenizer, model, output_path)

