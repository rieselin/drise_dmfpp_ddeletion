
import os
from ultralytics import YOLO
import torchvision
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from xai.drise_batch import DRISEBatch
from utils.utils import load_and_convert_bboxes
from utils.plot_utils import plot_image_with_bboxes, plot_saliency_and_targetbb_on_image
from datetime import datetime
import ssl
import os, re
from unsloth import FastVisionModel # FastLanguageModel for LLMs
from transformers import TextStreamer
from PIL import Image
import requests
from io import BytesIO



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
})


# ## Data Processing


#########################
# Import data
#########################
height, width = args.input_size
img_path = args.datadir + args.img_name
orig_img = Image.open(img_path + '.jpg')
resized_img = orig_img.resize((width, height), Image.LANCZOS)
img_np = np.array(resized_img)

plt.imshow(img_np)
print(img_np.shape)
print(img_np.shape, img_np.dtype)

# preprocessing function
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

tensor = preprocess(img_np)
tensor = tensor.unsqueeze(0).to(args.device) # 1,3,224,224





labels = args.annotations_dir + args.img_name + '.txt'
print('Labels directory:', labels)

date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

for tc in [0]:#[0,1,2,3,4,5,6,7]:
    bboxes, _ = load_and_convert_bboxes(labels,img_height=args.input_size[0],img_width=args.input_size[1], target_class= tc)
    plot_image_with_bboxes(img_np,bboxes, save_to=f'output/{args.img_name}_bboxes_class{tc}_{date_time}.png')


# ### set the target class


target_class = args.target_classes[0] # select a single class (list is given)
bboxes, _ = load_and_convert_bboxes(labels,img_height=args.input_size[0],img_width=args.input_size[1], target_class= target_class)
target_bbox = bboxes[0] # select the first bbox --> multiple might be given in the same image
plot_image_with_bboxes(img_np,[target_bbox], save_to=f'output/{args.img_name}_target_bbox_class{target_class}_{date_time}.png')
print('Target bbox:',target_bbox)


# ## YOLO
# Load model and test to see its predictions


model = YOLO(args.model_path, task='detect')


results=model.predict(tensor) 

boxes = results[0].boxes  # Assuming we have one image and accessing the first result
predicted_bboxes = []
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    # Convert to a list of (x, y) tuples
    bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    predicted_bboxes.append(bbox)
# print(bboxes)
image_with_bboxes = plot_image_with_bboxes(img_np, predicted_bboxes)


# ## D-RISE


# ### Generate masks


explainer = DRISEBatch(
    model=model, 
    input_size=args.input_size, 
    device=args.device,
    gpu_batch=args.gpu_batch
)

# if generate or load
generate_new = True

mask_filename = f'{args.mask_type}_n{args.N}_s{args.resolution}_p{args.p1}_{args.input_size[0]}x{args.input_size[1]}'
mask_path = args.maskdir + mask_filename + '.npy'
print(mask_path)

if generate_new or not os.path.isfile(mask_path):
    explainer.generate_masks_rise(N=args.N, s=args.resolution, p1=args.p1, savepath= mask_path)
else:
    explainer.load_masks(mask_path)
    print('Masks are loaded.')


# Visualize 3 generated masks


num_masks=3
masks = explainer.masks[:3]

masked_image = torch.mul(masks.to(args.device), tensor)
masked_image = masked_image.cpu().numpy()
print(masked_image.shape)

masks = masks.cpu()
if masks.ndim == 4:  # If the masks have a shape of (N, 1, H, W)
    masks = masks[:, 0, :, :]

fig, axes = plt.subplots(3, num_masks, figsize=(15, 10))

for i, ax in enumerate(axes[0]):
    ax.imshow(img_np)
    ax.axis('off')
    ax.set_title(f'Original image {i + 1}')

for i, ax in enumerate(axes[1]):
    ax.imshow(masks[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f'Mask {i + 1}')

for i, ax in enumerate(axes[2]):
    ax.imshow(masked_image[i].transpose(1, 2, 0))
    ax.axis('off')
    ax.set_title(f'Masked Image {i + 1}')
    
plt.show()


# ### Apply XAI


# print(tensor.permute(0,1,3,2).shape)
saliency = explainer(
    x=tensor,
    target_class_indices=args.target_classes,
    target_bbox=target_bbox
)


# ### Plot heatmap/saliency map given by XAI


plot_saliency_and_targetbb_on_image(
        height=args.input_size[0], width=args.input_size[1], 
        img_name=args.img_name, 
        img=img_np,
        saliency_map=saliency[target_class], 
        target_class_id= target_class,
        target_bbox=target_bbox,
        save_to=f'output/{args.img_name}_saliency_targetbb_class{target_class}_{date_time}.png'
    )


# ## do the same for another target bbox


bboxes, _ = load_and_convert_bboxes(labels,img_height=args.input_size[0],img_width=args.input_size[1], target_class= target_class)
target_bbox = bboxes[1] # select the first bbox --> multiple might be given in the same image
plot_image_with_bboxes(img_np,[target_bbox], save_to=f'output/{args.img_name}_predicted_bbox_{date_time}.png')
print('Target bbox:',target_bbox)


saliency = explainer(x=tensor,
                     target_class_indices=[target_class],
                     target_bbox=target_bbox)


image_with_bbox_and_saliency = plot_saliency_and_targetbb_on_image(
        height=args.input_size[0], width=args.input_size[1], 
        img_name=args.img_name, 
        img=img_np,
        saliency_map=saliency[target_class], 
        target_class_id= target_class,
        target_bbox=target_bbox
    )


# ## LLAMA VL




model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)



FastVisionModel.for_inference(model) # Enable for inference!



#yolo detected bboxes
yoloPredictedBboxes = Image.open(f'output/{args.img_name}_predicted_bbox_{date_time}.png')

# saliency map with target bbox
driseSaliency = Image.open(f'output/{args.img_name}_saliency_targetbb_class{target_class}_{date_time}.png')


images = [resized_img, yoloPredictedBboxes, driseSaliency]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

composed = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  composed.paste(im, (x_offset,0))
  x_offset += im.size[0]

composed.save(f'output/composedImage{target_class}_{date_time}.jpg')


instruction = """You are the Visual LLM specializing in detailed explanation chaining for Visual-LLMs.
You are provided with an image, the bounding box predicted by YOLO and a saliency map for one bounding box generated with DRISE.
Give a detailed analysis on Color, Shape and Result to explain how the model reached the bounding box.

Do not make up any information that is not present in the image, bounding boxes or saliency map.
Do not repeat the same information in different sections.
Do not explain any of the used models or concepts. Only focus on the given image, bounding boxes and saliency map.

"""




messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    composed,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")


text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 400,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


