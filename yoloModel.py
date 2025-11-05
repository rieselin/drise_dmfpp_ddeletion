from ultralytics import YOLO
from utils.plot_utils import plot_image_with_bboxes

class YoloModel:
    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.model_path, task='detect')
    
    def predict(self, tensor):
        results=self.model.predict(tensor) 
        return results
    def extract_bboxes(self, results):
        boxes = results[0].boxes  # Assuming we have one image and accessing the first result
        predicted_bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            # Convert to a list of (x, y) tuples
            bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            predicted_bboxes.append(bbox)
        return predicted_bboxes
    def plot_bboxes(self, img_np, predicted_bboxes, output_path):
        image_with_bboxes = plot_image_with_bboxes(img_np, predicted_bboxes, save_to=f'{output_path}yolo_predicted_bboxes.png', show_plot=self.args.show_plots, tight_save=self.args.remove_all_borders_and_legends_from_images)
        return image_with_bboxes