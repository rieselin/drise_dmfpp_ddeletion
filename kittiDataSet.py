import fiftyone as fo


import json
import os


class KittiDataSet:
    def __init__(self, dataset_name="kitti", split="train", max_samples=None, dataset_dir=None):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset_dir = dataset_dir
        self.max_samples = max_samples
        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        fo.config.dataset_zoo_dir = self.dataset_dir
        dataset = fo.zoo.load_zoo_dataset(
            self.dataset_name,
            split=self.split,
            max_samples=self.max_samples
        )
        return dataset
    def save_bboxes_to_file(self, overwrite=False):
        # save bboxes in class_id x1 y1 x2 y1 x2 y2 x1 y2 (as text file imagename.txt)
        output_dir = self.dataset_dir +"/"+ self.dataset_name + "/" + self.split + "/"
        annotations_dir = output_dir + "annotations/"

        os.makedirs(annotations_dir, exist_ok=True)

        # check if labelfiles already exist
        if not overwrite:
            existing_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
            if len(existing_files) >= self.max_samples:
                print(f"Label files already exist in '{annotations_dir}'. Skipping save.")
                return

        for sample in self.dataset:
            # Get detections
            detections = sample.ground_truth.detections if sample.ground_truth else []

            label_lines = []
            for det in detections:
                label = det.label
                class_id = class_map.get(label, 0)

                # FiftyOne bounding boxes are [x, y, w, h] relative to image size (0–1)
                x_rel, y_rel, w_rel, h_rel = det.bounding_box

                # Compute corner coordinates (still relative)
                x1 = x_rel
                y1 = y_rel
                x2 = x_rel + w_rel
                y2 = y_rel + h_rel

                # Format: class_id x1 y1 x2 y1 x2 y2 x1 y2
                label_line = f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
                label_lines.append(label_line)

            # Save to file with same name as image
            img_name = os.path.splitext(os.path.basename(sample.filepath))[0]
            label_path = os.path.join(annotations_dir, f"{img_name}.txt")

            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

        print(f"Saved bounding box files to '{annotations_dir}'.")



# Define mapping from class name to numeric ID
class_map = {
    "DontCare": 0,
    "Pedestrian": 1,
    "Car": 2,
    "Cyclist": 3,
    "Van": 4,
    "Truck": 5,
    "Misc": 6,
    "Person_sitting": 7,
    "Tram": 8,
}

# # Load COCO annotations
# with open("labels.json", "r") as f:
#     data = json.load(f)

# images = {img["id"]: img for img in data["images"]}

# # Create output directory
# os.makedirs("bbox_labels", exist_ok=True)

# # Group annotations by image_id
# image_annotations = {}
# for ann in data["annotations"]:
#     image_annotations.setdefault(ann["image_id"], []).append(ann)

# # Process each image
# for image_id, anns in image_annotations.items():
#     img = images[image_id]
#     width, height = img["width"], img["height"]
#     file_name = os.path.splitext(img["file_name"])[0]
#     anns_sorted = sorted(anns, key=lambda a: a["category_id"])

#     lines = []
#     for ann in anns_sorted:
#         cat_id = ann["category_id"]
#         x, y, w, h = ann["bbox"]

#         # normalize
#         x_min, y_min = x / width, y / height
#         x_max, y_max = (x + w) / width, (y + h) / height

#         # COCO bbox is rectangular, your example uses polygons,
#         # so we output 4 corner points (clockwise)
#         line = f"{cat_id} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} {x_min:.6f} {y_max:.6f}"
#         lines.append(line)

#     with open(f"bbox_labels/{file_name}.txt", "w") as f:
#         f.write("\n".join(lines))
