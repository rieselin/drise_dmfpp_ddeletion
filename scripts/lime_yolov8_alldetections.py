from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
import torchvision
import argparse

# Inherits from scikit-image: https://github.com/marcotcr/lime/blob/master/lime/wrappers/scikit_image.py
from lime import lime_image
# Check segmentation algorithms: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
from lime.wrappers.scikit_image import SegmentationAlgorithm 
from utils.lime_utils import get_probab_class_wrapper
from utils.utils import normalize_bboxes
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


#########################
# Arguments
#########################
parser = argparse.ArgumentParser()
# hw related
parser.add_argument("--device", default='cuda:1')
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
# mask generation
parser.add_argument("--target_classes", default="0")
parser.add_argument("--num_classes", type=int, default=2)

args = parser.parse_args()
args.target_classes = [int(cls) for cls in args.target_classes.split(',')] if args.target_classes != "" else [1]

#########################
# Define Model
#########################
model = YOLO(args.model_path, task='detect').to(args.model_device)

#########################
# Generate Explainer Instance
#########################
explainer = lime_image.LimeImageExplainer() #https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image
segmenter = SegmentationAlgorithm('slic',n_segments=100) 

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

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    tensor = preprocess(resized_img)
    tensor = tensor.unsqueeze(0).to(args.device)
    
    # labels = args.labels_dir + img_name + '.txt'
    # print('Labels directory:', labels)
    
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
            explanation = explainer.explain_instance(
                image=img_np, 
                classifier_fn=get_probab_class_wrapper(img_np=img_np,model=model, target_class=_target_class, num_classes=int(args.num_classes)), # classification function
                top_labels=1, #produce explanations for the K labels with highest prediction probabilities
                hide_color=0, 
                num_samples=1000, # number of images that will be sent to classification function
                segmentation_fn = segmenter, # SegmentationAlgorithm, wrapped skimage
                # batch_size=100,
            ) 
            

            # Obtener el label y los superpíxeles
            segments = explanation.segments

            # Crear un heatmap con los pesos de los superpíxeles
            heatmap = np.zeros(shape=(img_np.shape[0], img_np.shape[1]))

            # Iterar sobre los pesos locales de los superpíxeles
            for superpixel, weight in explanation.local_exp[_target_class]:
                # Asignar el peso a los píxeles del superpíxel correspondiente
                heatmap[segments == superpixel] = weight

            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            
            # save saliency map
            # ensure target directory exist
            os.makedirs(args.saliency_map_dir, exist_ok=True)
            # set filename/path
            model_type = args.model_path.split(os.sep)[-3]
            filename = f'img{img_name}_m_{model_type}_d_'+ f'_class_{_target_class}_bb_{target_norm_bbox}.npy'
            saliency_map_path = os.path.join(args.saliency_map_dir, filename)
            # save it
            np.save(saliency_map_path, heatmap)
