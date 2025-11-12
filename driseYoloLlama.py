
import ssl
import os
import json

from cocoDataSet import CocoDataSet
from dataProcessing import DataProcessing
from driseExplainer import DRISEExplainer
from kittiDataSet import KittiDataSet
from llamaVisionModel import LLMAVisionModel
from metrics.utils import calculate_iou
from utils.plot_utils import plot_image_with_bboxes
from yoloModel import YoloModel
from datetime import datetime
from PIL import Image

from kittiDataSet import class_map_rev


ssl._create_default_https_context = ssl._create_unverified_context

class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
args = Args(**{
    'img_names':['000000', '000001', '000002','000003', '000004', '000005', '000006', '000007', '000008', '000009'],#  ['00901'],# , #[ '000001'],#['000000000419', '000000000260', '000000000328', '000000000149', '000000000722', '000000000730'],
    'model_path': 'train13/weights/best.pt', #yolov8n.pt',#'use_case/models/best.pt',#
    'datadir': 'kitti_data/kitti/train/data/',#'use_case/',#,#,#'kitti_data/kitti/train/data/', #'coco_data/coco-2017/train/data/',
    'annotations_dir': 'kitti_data/kitti/train/annotations/',#'use_case/',#'kitti_data/kitti/train/annotations/',#'use_case/',#'kitti_data/kitti/train/annotations/', #'coco_data/coco-2017/train/data/',
    'device': 'cuda:0',
    'input_size': (480, 800),
    'gpu_batch': 16,
    'mask_type': 'mfpp', # mfpp or rise
    'maskdir': 'masks/',
    'N': 1000,
    'resolution': 8,
    'p1': 0.5,
    'target_classes': [1,2,3,0],
    'show_plots': False,
    'run_id_tag': '',
    'instruction': '''describe why the object detection model made the bounding box prediciton based on the saliency map and the predicted bounding box on the image. 
    if there is no bounding box on the image, explain based on the bounding box why the model did not detect the object.
    the saliency map colors are from blue (negative contribution) over green 0.9 (no contribution) to red (positive contribution)
    do not explain any of the concepts in the instruction. 
    keep the answer concise within 100 words.''',
    'run_only_first_bbox': False,
# send to llama configs
    'send_saliency_map': True,
    'send_labelled_bbox': False,
    'send_predicted_bbox': True, 
    'send_all_bboxes_of_image_at_once': False,
    'remove_all_borders_and_legends_from_images': True,
    'send_weird_test': False
})


kittiDataSet = KittiDataSet(
    dataset_name="kitti", 
    split="train", 
   # classes=["Car", "Pedestrian"], 
    max_samples=10, 
    dataset_dir='kitti_data/')
kittiDataSet.save_bboxes_to_file()


cocoDataSet = CocoDataSet(
    dataset_name="coco-2017", 
    split="train", 
    classes=["person", "car"], 
    max_samples=10, 
    dataset_dir='coco_data/')

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

    composedImagePaths = []
    

    for target_class in args.target_classes:
        #target_class = dataProcessing.set_target_class()

        bboxes = dataProcessing.plot_bboxes(img_np, labels, target_class)
        if len(bboxes) == 0:
            print(f"No bounding boxes found for target class {target_class} in image {args.img_name}. Skipping to next class.")
            continue

        # ## YOLO
        yoloModel = YoloModel(args)
        results = yoloModel.predict(img_np)
        predicted_bboxes = yoloModel.extract_bboxes(results)
        if len(predicted_bboxes) == 0:
            print(f"No predicted bounding boxes found by YOLO in image {args.img_name}.")
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

            try:
                # Store original setting
                original_send_pred_bbox = args.send_predicted_bbox
                original_instruction = args.instruction

                # find the predicted bbox (in predicted_bboxes) that matches the bbox given in the labels
                # Find the predicted bbox that best matches the ground-truth bbox
                best_iou = 0
                best_pred_bbox = None

                for pred_bbox in predicted_bboxes:
                    iou = calculate_iou(bbox, pred_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_bbox = pred_bbox

                if best_iou < 0.3 or len(predicted_bboxes) == 0:
                    print(f"No good match found for bbox {bbox} (best IoU: {best_iou:.2f})")
                    args.instruction = args.instruction + f'The model did not detect the object with target class {class_map_rev[target_class]}'
                    args.send_predicted_bbox = False
                
                saliency = driseExplainer.apply_saliency(tensor, bbox)
                # plot salicney map with both target bbox and predicted bbox

                if args.send_weird_test:
                    # image with labelled bbox
                    composed = plot_image_with_bboxes(img_np, [bbox], show_plot=args.show_plots, save_to=f'{output_path_saliency}bbox_{i}_tc_{target_class}.png', tight_save=args.remove_all_borders_and_legends_from_images)
                    composed1 = Image.open( f'{output_path_saliency}bbox_{i}_tc_{target_class}.png')
                    # image with predicted bbox and saliency map
                    driseExplainer.plot_saliency(saliency, None, best_pred_bbox, img_np, f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png', target_class)
                    composed2 = Image.open( f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png') 

                    images = [composed1, composed2]

                    widths, heights = zip(*(i.size for i in images))

                    total_width = sum(widths)
                    max_height = max(heights)

                    composed = Image.new('RGB', (total_width, max_height))

                    x_offset = 0
                    for im in images:
                        composed.paste(im, (x_offset,0))
                        x_offset += im.size[0]

                    composed.save(f'{output_path}composed_image_{i}_tc_{target_class}.jpg')
                    composed.filename = f'{output_path}composed_image.jpg'

                    args.instruction = args.instruction + f'The image shows on the left the labelled bounding box and on the right the predicted bounding box with the saliency map overlayed.'


                elif not args.send_saliency_map and not args.send_labelled_bbox and not args.send_predicted_bbox:
                    composed = resized_img
                elif not args.send_saliency_map and args.send_labelled_bbox and not args.send_predicted_bbox:
                    composed = plot_image_with_bboxes(img_np, [bbox], show_plot=args.show_plots, save_to=f'{output_path_saliency}bbox_{i}_tc_{target_class}.png', tight_save=args.remove_all_borders_and_legends_from_images)
                    composed = Image.open( f'{output_path_saliency}bbox_{i}_tc_{target_class}.png')
                elif not args.send_saliency_map and not args.send_labelled_bbox and args.send_predicted_bbox:
                    composed = plot_image_with_bboxes(img_np, [best_pred_bbox], show_plot=args.show_plots, save_to=f'{output_path_saliency}bbox_{i}_tc_{target_class}.png', tight_save=args.remove_all_borders_and_legends_from_images)
                    composed = Image.open( f'{output_path_saliency}bbox_{i}_tc_{target_class}.png')
                elif not args.send_saliency_map and args.send_labelled_bbox and args.send_predicted_bbox:
                    composed = plot_image_with_bboxes(img_np, [bbox, best_pred_bbox], show_plot=args.show_plots, save_to=f'{output_path_saliency}bbox_{i}_tc_{target_class}.png', tight_save=args.remove_all_borders_and_legends_from_images)
                    composed = Image.open( f'{output_path_saliency}bbox_{i}_tc_{target_class}.png')
                # send saliency map
                elif args.send_saliency_map and not args.send_labelled_bbox and not args.send_predicted_bbox:
                    driseExplainer.plot_saliency(saliency, None, None, img_np, f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png', target_class)
                    composed = Image.open( f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png') 
                elif args.send_saliency_map and args.send_labelled_bbox and not args.send_predicted_bbox:
                    driseExplainer.plot_saliency(saliency, bbox, None, img_np, f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png', target_class)
                    composed = Image.open( f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png') 
                elif args.send_saliency_map and not args.send_labelled_bbox and args.send_predicted_bbox:
                    driseExplainer.plot_saliency(saliency, None, best_pred_bbox, img_np, f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png', target_class)
                    composed = Image.open( f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png') 
                elif args.send_saliency_map and args.send_labelled_bbox and args.send_predicted_bbox:
                    driseExplainer.plot_saliency(saliency, bbox, best_pred_bbox, img_np, f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png', target_class)
                    composed = Image.open( f'{output_path_saliency}drise_saliency_{i}_tc_{target_class}.png') 
                



                
                composedImagePaths.append(composed.filename)

                

                if not args.send_all_bboxes_of_image_at_once:
                    #composed = llamaVisionModel.createImageInput(output_path, resized_img, target_class)
                    inputs = llamaVisionModel.compose_input(tokenizer, composed)
                    llamaVisionModel.generate_response(inputs, tokenizer, model, f'{output_path_llama}llama_response{i}_tc_{target_class}.txt')
                
                i += 1
            except Exception as e:
                print(f"Error processing bbox {bbox}: {e}")
                continue
            finally:
                # Restore original setting for next bbox
                args.send_predicted_bbox = original_send_pred_bbox
                args.instruction = original_instruction

        


    if (len(composedImagePaths) == 0):
        print(f"Skipping metadata generation for image {args.img_name} due to previous errors or no composed images.")
        continue
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
            img_path for img_path in composedImagePaths
        ],
        "llama_responses": [
            f"{output_path_llama}llama_response{j}.txt"
            for j in range(i)
        ]
    }

    if args.send_all_bboxes_of_image_at_once:

        images = meta_data["composed_images"]

        if len(images) == 0:
            print("No composed images to process for LLaMA.")
            continue

        images = [Image.open(img_path) for img_path in images]

        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        composed = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            composed.paste(im, (x_offset,0))
            x_offset += im.size[0]

        composed.save(f'{output_path}composed_image.jpg')
        meta_data["composed_images"] = f'{output_path}composed_image.jpg'

        inputs = llamaVisionModel.compose_input(tokenizer, composed)
        llamaVisionModel.generate_response(inputs, tokenizer, model, f'{output_path_llama}llama_response_all_bboxes.txt')
        meta_data["llama_responses"] = f'{output_path_llama}llama_response_all_bboxes.txt'


    with open(meta_output_path, 'w') as f:
        # add image name, target classes, run id tag, instruction, send to llama configs, add composed image path
        
        # Write JSON to file
        json.dump(meta_data, f, indent=4)

        
