
import os
from utils.plot_utils import plot_saliency_and_targetbb_on_image
from xai.drise_batch import DRISEBatch

class DRISEExplainer:
    def __init__(self, args, yoloModel, generate_new=True):
        self.args = args
        self.generate_new = generate_new
        self.yoloModel = yoloModel
        self.explainer = DRISEBatch(
            model= self.yoloModel.model, 
            input_size=args.input_size, 
            device=args.device,
            gpu_batch=args.gpu_batch
        )
    def generate_masks(self):
        mask_filename = f'{self.args.mask_type}_n{self.args.N}_s{self.args.resolution}_p{self.args.p1}_{self.args.input_size[0]}x{self.args.input_size[1]}'
        mask_path = self.args.maskdir + mask_filename + '.npy'
        print(mask_path)

        if self.generate_new or not os.path.isfile(path=mask_path):
            self.explainer.generate_masks_rise(N=self.args.N, s=self.args.resolution, p1=self.args.p1, savepath= mask_path)
        else:
            self.explainer.load_masks(mask_path)
            print('Masks are loaded.')
        return mask_path
    def apply_saliency(self, tensor, target_bbox):
        saliency = self.explainer(
            x=tensor,
            target_class_indices=self.args.target_classes,
            target_bbox=target_bbox
        )
        return saliency
    def plot_saliency(self, saliency, target_bbox, img_np, output_path, target_class):
        plot_saliency_and_targetbb_on_image(
            height=self.args.input_size[0], width=self.args.input_size[1], 
            img_name=self.args.img_name, 
            img=img_np,
            saliency_map=saliency[target_class], 
            target_class_id= target_class,
            target_bbox=target_bbox,
            show_plot = self.args.show_plots,
            save_to=f'{output_path}drise_saliency.png'
        )