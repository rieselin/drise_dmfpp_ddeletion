import argparse
import os
import sys
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torchvision
import csv
import ast
# Import your custom modules
from metrics.deletion import CausalMetric
from metrics.pointing_game import PointingGame
from metrics.utils import correspond_box, minimal_subset
from utils.utils import extract_bbox_class_and_path, scale_bbox, load_and_convert_bboxes
from utils.plot_utils import plot_image_with_bboxes, plot_saliency_and_targetbb_on_image

def parse_args():
    parser = argparse.ArgumentParser(description='Script to evaluate saliency maps with different metrics.')

    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLO model file.')
    parser.add_argument('--datadir', type=str, required=True, help='Directory containing the dataset images.')
    parser.add_argument('--labels_dir', type=str, required=True, help='Directory containing the label files.')
    parser.add_argument('--saliency_map_dir', type=str, required=True, help='Directory containing the saliency map files.')
    parser.add_argument('--csv_dir', type=str, required=True, help='Path to save the CSV results.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the dataset.')
    # parser.add_argument('--input_size', type=int, nargs=2, default=[480, 640], help='Input size for the images.')
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    height, width = args.height, args.width
    
    # Filter the images at the data folder
    img_available = [f for f in os.listdir(args.datadir) if f.endswith('.jpg')]
    img_available_no_ext = [os.path.splitext(f)[0] for f in img_available]
    for i in img_available:
        print(i)
    # Filter the saliency maps available
    saliency_files = [f for f in os.listdir(args.saliency_map_dir) if f.endswith('.npy')]
    for s in saliency_files:
        print(s)

    # load model
    model = YOLO(args.model_path, task='detect').to(args.device)

    # define metrics
    deletion = CausalMetric(
        height=height, width=width,
        model=model,
        mode='del',
        step=1000,
        device=args.device,
    )
    pointing_game = PointingGame()

    
    # initialize dictionaries of metrics
    auc_scores = {k:[] for k in range(args.num_classes)} #number of objects
    minimal_subset_scores = {k:[] for k in range(args.num_classes)} #number of objects
    d_auc_scores = {k:[] for k in range(args.num_classes)} #number of objects
    d_minimal_subset_scores = {k:[] for k in range(args.num_classes)} #number of objects
    csv_data = []

    # begin metrics calculation for-loop
    for _i, img_name in enumerate(sorted(img_available_no_ext)):
        print('\n{}/{} - Image {}'.format(_i, len(img_available_no_ext)-1, img_name))
        img_path = os.path.join(args.datadir, img_name + '.jpg')
        # Load and preprocess image
        orig_img = Image.open(img_path)
        # resize (if needed)
        resized_img = orig_img.resize((width, height), Image.LANCZOS)
        img_np = np.array(resized_img)
        # totensor
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        tensor = preprocess(resized_img).unsqueeze(0).to(args.device)

        # Ground truth labels (for Pointing Game)
        labels = os.path.join(args.labels_dir, img_name + '.txt')
        print('Labels directory:', labels)

        # Get the associated explanations
        matching_saliency_files = [s for s in saliency_files if f"img{img_name}_" in s]

        for _j, saliency_file in enumerate(sorted(matching_saliency_files)):
            # Extract metadata (class and bounding box)  --> Split the string to extract the class value
            parts = saliency_file.split('class_')[1]
            target_class = int(parts.split('_bb')[0])
            print('Objects {}|{} - Class {}'.format(_j, len(matching_saliency_files)-1, target_class))

            # Load the saliency map
            saliency_map = np.load(os.path.join(args.saliency_map_dir, saliency_file))
            flag_only_zeros = True if np.all(saliency_map == 0) else False

            # get BB and scale it (needed for PG metric)
            norm_target_bbox = parts.split('bb_')[1].replace('.npy', '')
            if isinstance(norm_target_bbox, str):
                norm_target_bbox = ast.literal_eval(norm_target_bbox)
            target_bbox = scale_bbox(norm_target_bbox, height, width)

            # Forward pass through the model
            results = model(tensor, verbose=False)
            # check if the model detects that object (might be 0 if targetBB was the GT)
            flag_detects_obj = 0
            initial_score = 0
            for result in results:
                for box in result.boxes:
                    score = float(box.conf.item())
                    label = int(box.cls.item())
                    if score > 0.7 and label == target_class:
                        flag_detects_obj = 1
                        initial_score = score if score > initial_score else initial_score # store max

            ######################################
            # ***Deletion
            ######################################
            scores, auc = deletion.single_run(
                img_tensor=tensor,
                explanation=saliency_map,
                target_class_index=target_class,
                verbose=0,
            )
            auc_scores[target_class].append(auc)
            min_subset_score = 100 * minimal_subset(scores, k_threshold=0.7)
            minimal_subset_scores[target_class].append(min_subset_score)

            ######################################
            # ***D-Del --> Deletion with Detection
            ######################################
            d_scores, d_auc = deletion.single_run(
                img_tensor=tensor,
                explanation=saliency_map,
                target_class_index=target_class,
                verbose=0,
                target_bbox=target_bbox
            )
            d_auc_scores[target_class].append(d_auc)
            d_minimal_subset_score = 100 * minimal_subset(d_scores, k_threshold=0.7)
            d_minimal_subset_scores[target_class].append(d_minimal_subset_score)
            
            ######################################
            # ***Pointing Game - PG
            ######################################
            # Associate ground truth boxes
            gt_scaled_bboxes, gt_normalized_bboxes, gt_class_bboxes = load_and_convert_bboxes(
                labels,
                img_height=height,
                img_width=width,
                target_class=target_class,
                return_class=True
            )
            gt_boxes, _ = correspond_box(
                predictbox=[target_bbox],
                predict_classes=[target_class],
                groundtruthboxes=gt_scaled_bboxes,
                groundtruth_classes=gt_class_bboxes
            )

            # ensure only 1 ground truth bounding box is returned (one per image)
            if len(gt_boxes) == 1:
                is_within_bbox, salient_pixel = pointing_game.forward(
                    img_np=img_np,
                    heatmap=saliency_map,
                    target_bbox=gt_boxes[0], # gt_boxes expects to return a list for the multiple inputs (we only pass one image; thus index by [0])
                )
                activation_coverage = pointing_game.get_activation_coverage(
                    heatmap=saliency_map,
                    target_bbox=gt_boxes[0] # gt_boxes expects to return a list for the multiple inputs (we only pass one image; thus index by [0])
                )
            else:
                print('No BB for that detection!')
                is_within_bbox = -1
                activation_coverage = -1

            # Append data to CSV
            csv_data.append([
                img_name, target_class, initial_score, flag_detects_obj, int(flag_only_zeros),
                auc, d_auc, min_subset_score, d_minimal_subset_score,
                int(is_within_bbox), activation_coverage
            ])

    # Save to CSV file
    with open(args.csv_dir, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        # Write the header
        writer.writerow(['img_name', 'target_class', 'initial_score', 'detects_obj', 'only_zeros', 'auc', 'd-auc', 'minimalsubset', 'd-minimalsubset', 'pg', 'coverage'])
        # Write the data
        writer.writerows(csv_data)

    print(f"Results saved to {args.csv_dir}")
