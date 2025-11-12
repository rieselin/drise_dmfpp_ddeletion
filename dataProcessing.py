import torchvision
import numpy as np
from PIL import Image
from utils.utils import load_and_convert_bboxes
from utils.plot_utils import plot_image_with_bboxes


class DataProcessing:
    def __init__(self, args, output_path):
        self.args = args
        self.output_path = output_path
    def import_data(self):
        height, width = self.args.input_size
        img_path = self.args.datadir + self.args.img_name
        if self.args.datadir.startswith('kitti'):
            orig_img = Image.open(img_path + '.png')
        else: 
            orig_img = Image.open(img_path + '.jpg')
        self.args.input_size = (height, width)

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
    def plot_bboxes(self, img_np, labels, target_class):
        bboxes, _ = load_and_convert_bboxes(labels,img_height=self.args.input_size[0],img_width=self.args.input_size[1], target_class= target_class)
        if len(bboxes)==0:
            # don't plot anything
            return bboxes
        plot_image_with_bboxes(img_np,bboxes, save_to=f'{self.output_path}bboxes_tc_{target_class}.png', show_plot=self.args.show_plots, tight_save=self.args.remove_all_borders_and_legends_from_images)
        return bboxes
    