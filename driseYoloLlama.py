
import ssl

from dataProcessing import DataProcessing
from driseExplainer import DRISEExplainer
from llamaVisionModel import LLMAVisionModel
from yoloModel import YoloModel
from datetime import datetime


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
    'instruction': """You are the Visual LLM specializing in detailed explanation chaining for Visual-LLMs.
You are provided with an image, the bounding box predicted by YOLO and a saliency map for one bounding box generated with DRISE.
Give a detailed analysis on Color, Shape and Result to explain how the model reached the bounding box.

Do not make up any information that is not present in the image, bounding boxes or saliency map.
Do not repeat the same information in different sections.
Do not explain any of the used models or concepts. Only focus on the given image, bounding boxes and saliency map.

"""
})

# ## Data Processing
date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

dataProcessing = DataProcessing(args, date_time)
resized_img, img_np = dataProcessing.import_data()
tensor = dataProcessing.preprocess_image(img_np)
labels = dataProcessing.load_labels()
target_class = dataProcessing.set_target_class()
bboxes = dataProcessing.plot_bboxes(img_np, labels, target_class)
target_bbox = dataProcessing.plot_target_bbox(img_np, bboxes, target_class)


# ## YOLO
# Load model and test to see its predictions
yoloModel = YoloModel(args)
results = yoloModel.predict(tensor)
predicted_bboxes = yoloModel.extract_bboxes(results)
image_with_bboxes = yoloModel.plot_bboxes(img_np, predicted_bboxes, date_time=date_time)


# ## D-RISE
driseExplainer = DRISEExplainer(args, yoloModel, generate_new=True)
mask_path = driseExplainer.generate_masks()
saliency = driseExplainer.apply_saliency(tensor, target_bbox)
driseExplainer.plot_saliency(saliency, target_bbox, img_np, date_time, target_class)


# ## LLAMA VL
llamaVisionModel = LLMAVisionModel(args)
model, tokenizer = llamaVisionModel.load_model()
composed = llamaVisionModel.createImageInput(date_time, resized_img, target_class)
inputs = llamaVisionModel.compose_input(tokenizer, composed)
llamaVisionModel.generate_response(inputs, tokenizer, model)

