
import ssl
import os
import json

from cocoDataSet import CocoDataSet
from dataProcessing import DataProcessing
from driseExplainer import DRISEExplainer
from llamaVisionModel import LLMAVisionModel
from metrics.utils import calculate_iou
from utils.plot_utils import plot_image_with_bboxes
from yoloModel import YoloModel
from datetime import datetime
from PIL import Image


ssl._create_default_https_context = ssl._create_unverified_context

class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
args = Args(**{
    'img_names': ['000000000419', '000000000260', '000000000328', '000000000149', '000000000722', '000000000730'],
    'model_path': 'use_case/models/best.pt',
    'datadir': 'coco_data/coco-2017/train/data/',
    'annotations_dir': 'coco_data/coco-2017/train/data/',
    'device': 'cuda:0',
    'input_size': (480, 640),
    'gpu_batch': 16,
    'mask_type': 'rise',
    'maskdir': 'masks/',
    'N': 1000,
    'resolution': 8,
    'p1': 0.5,
    'target_classes': [1],
    'show_plots': False,
    'run_id_tag': '',
    'instruction': '',
    'run_only_first_bbox': False,
# send to llama configs
    'send_saliency_map': True,
    'send_labelled_bbox': False,
    'send_predicted_bbox': True
    
})


cocoDataSet = CocoDataSet(
    dataset_name="coco-2017", 
    split="train", 
    classes=["person", "car"], 
    max_samples=10, 
    dataset_dir='coco_data/')
cocoDataSet.load_dataset()

cocoDataSet.save_bboxes_to_file()

# overwrite args.input_size based on dataset images

date_time_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if args.run_id_tag != '':
        date_time_tag += f'_{args.run_id_tag}'

for args.img_name in args.img_names:

    # setup
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

    output_path_llama = f'{output_path}llama/'
    os.makedirs(output_path_llama, exist_ok=True)


    llamaVisionModel = LLMAVisionModel(args)
    model, tokenizer = llamaVisionModel.load_model()

    # todo : think about more sensible way to label instead of just increasing i
    # todo make run for and multiple images
    i = 0
    if args.run_only_first_bbox:
        bboxes = [bboxes[0]]
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

        # todo : do something if no match found
        if best_iou < 0.3:
            print(f"No good match found for bbox {bbox} (best IoU: {best_iou:.2f})")
            continue

        
        saliency = driseExplainer.apply_saliency(tensor, bbox)
        # plot salicney map with both target bbox and predicted bbox

        if not args.send_saliency_map and not args.send_labelled_bbox and not args.send_predicted_bbox:
            composed = resized_img
        elif not args.send_saliency_map and args.send_labelled_bbox and not args.send_predicted_bbox:
            composed = plot_image_with_bboxes(img_np, [bbox], show_plot=args.show_plots, save_to=f'{output_path_saliency}bbox_{i}.png')
            composed = Image.open( f'{output_path_saliency}bbox_{i}.png')
        elif not args.send_saliency_map and not args.send_labelled_bbox and args.send_predicted_bbox:
            composed = plot_image_with_bboxes(img_np, [best_pred_bbox], show_plot=args.show_plots, save_to=f'{output_path_saliency}bbox_{i}.png')
            composed = Image.open( f'{output_path_saliency}bbox_{i}.png')
        elif not args.send_saliency_map and args.send_labelled_bbox and args.send_predicted_bbox:
            composed = plot_image_with_bboxes(img_np, [bbox, best_pred_bbox], show_plot=args.show_plots, save_to=f'{output_path_saliency}bbox_{i}.png')
            composed = Image.open( f'{output_path_saliency}bbox_{i}.png')
        # send saliency map
        elif args.send_saliency_map and not args.send_labelled_bbox and not args.send_predicted_bbox:
            driseExplainer.plot_saliency(saliency, None, None, img_np, f'{output_path_saliency}drise_saliency_{i}.png', target_class)
            composed = Image.open( f'{output_path_saliency}drise_saliency_{i}.png') 
        elif args.send_saliency_map and args.send_labelled_bbox and not args.send_predicted_bbox:
            driseExplainer.plot_saliency(saliency, bbox, None, img_np, f'{output_path_saliency}drise_saliency_{i}.png', target_class)
            composed = Image.open( f'{output_path_saliency}drise_saliency_{i}.png') 
        elif args.send_saliency_map and not args.send_labelled_bbox and args.send_predicted_bbox:
            driseExplainer.plot_saliency(saliency, None, best_pred_bbox, img_np, f'{output_path_saliency}drise_saliency_{i}.png', target_class)
            composed = Image.open( f'{output_path_saliency}drise_saliency_{i}.png') 
        elif args.send_saliency_map and args.send_labelled_bbox and args.send_predicted_bbox:
            driseExplainer.plot_saliency(saliency, bbox, best_pred_bbox, img_np, f'{output_path_saliency}drise_saliency_{i}.png', target_class)
            composed = Image.open( f'{output_path_saliency}drise_saliency_{i}.png') 

        #composed = llamaVisionModel.createImageInput(output_path, resized_img, target_class)
        inputs = llamaVisionModel.compose_input(tokenizer, composed)
        llamaVisionModel.generate_response(inputs, tokenizer, model, f'{output_path_llama}llama_response{i}.txt')

        i += 1


    # save meta
    meta_output_path = f'{output_path}meta.json'

    # Collect metadata about the current run
    meta_data = {
        "date_time_tag": date_time_tag,
        "image_name": args.img_name,
        "model_path": args.model_path,
        "target_classes": args.target_classes,
        "run_id_tag": args.run_id_tag,
        "instruction": args.instruction,
        "send_to_llama": {
            "send_saliency_map": args.send_saliency_map,
            "send_labelled_bbox": args.send_labelled_bbox,
            "send_predicted_bbox": args.send_predicted_bbox
        },
        "output_paths": {
            "output_root": output_path,
            "saliency_dir": output_path_saliency,
            "llama_dir": output_path_llama,
        },
        "composed_images": [
            f"{output_path_saliency}drise_saliency_{j}.png" if args.send_saliency_map
            else f"{output_path_saliency}bbox_{j}.png"
            for j in range(i)
        ],
        "llama_responses": [
            f"{output_path_llama}llama_response{j}.txt"
            for j in range(i)
        ]
    }



    with open(meta_output_path, 'w') as f:
        # add image name, target classes, run id tag, instruction, send to llama configs, add composed image path
        
        # Write JSON to file
        json.dump(meta_data, f, indent=4)

    
