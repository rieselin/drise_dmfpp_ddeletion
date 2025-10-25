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
    def plot_bboxes(self, img_np, predicted_bboxes, date_time):
        image_with_bboxes = plot_image_with_bboxes(img_np, predicted_bboxes, save_to=f'output/{self.args.img_name}_predicted_bboxes_{date_time}.png')
        return image_with_bboxes
    def plot_target_bbox(self, img_np, target_bbox, date_time):
        image_with_target_bbox = plot_image_with_bboxes(img_np, [target_bbox], save_to=f'output/{self.args.img_name}_predicted_targetbbox_{date_time}.png')
        return image_with_target_bbox