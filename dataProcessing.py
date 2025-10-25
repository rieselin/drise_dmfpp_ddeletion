import torchvision
import numpy as np
from PIL import Image
from utils.utils import load_and_convert_bboxes
from utils.plot_utils import plot_image_with_bboxes


class DataProcessing:
    def __init__(self, args, date_time):
        self.args = args
        self.date_time = date_time
    def import_data(self):
        height, width = self.args.input_size
        img_path = self.args.datadir + self.args.img_name
        orig_img = Image.open(img_path + '.jpg')
        resized_img = orig_img.resize((width, height), Image.LANCZOS)
        img_np = np.array(resized_img)
        return resized_img, img_np

        
    def preprocess_image(self, img_np):
        # preprocessing function
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        tensor = preprocess(img_np)
        tensor = tensor.unsqueeze(0).to(self.args.device) # 1,3,224,224
        return tensor
    def load_labels(self):
        labels = self.args.annotations_dir + self.args.img_name + '.txt'
        return labels
    def set_target_class(self):
        target_class = self.args.target_classes[0] # select a single class (list is given)
        return target_class
    def plot_bboxes(self, img_np, labels, target_class):
        bboxes, _ = load_and_convert_bboxes(labels,img_height=self.args.input_size[0],img_width=self.args.input_size[1], target_class= target_class)
        plot_image_with_bboxes(img_np,bboxes, save_to=f'output/{self.args.img_name}_bboxes_class{target_class}_{self.date_time}.png')
        return bboxes
    def plot_target_bbox(self, img_np, bboxes, target_class):
        target_bbox = bboxes[0] # select the first bbox --> multiple might be given in the same image
        plot_image_with_bboxes(img_np,[target_bbox], save_to=f'output/{self.args.img_name}_target_bbox_class{target_class}_{self.date_time}.png')
        print('Target bbox:',target_bbox)
        return target_bbox
    