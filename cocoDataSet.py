import fiftyone as fo


import json
import os

# name = "my-dataset"
# dataset_dir = "/path/to/detection-dataset"

# Create the dataset
# dataset = fo.Dataset.from_dir(
#     dataset_dir=dataset_dir,
#     dataset_type=fo.types.COCODetectionDataset, # Change with your type
#     name=name,
# )

class CocoDataSet:
    def __init__(self, dataset_name="coco-2017", split="train", classes=None, max_samples=None, dataset_dir=None):
        self.dataset_name = dataset_name
        self.split = split
        self.classes = classes
        self.max_samples = max_samples
        self.dataset_dir = dataset_dir
        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        fo.config.dataset_zoo_dir = self.dataset_dir
        dataset = fo.zoo.load_zoo_dataset(
            self.dataset_name,
            split=self.split, # can be "train", "validation", or "test"
            label_types=["detections"],
            classes=self.classes,
            max_samples=self.max_samples
        )
        return dataset
    def save_bboxes_to_file(self):
        # save bboxes in class_id x_min y_min x_max y_max format (as text file imagename.txt)
        output_dir = self.dataset_dir +"/"+ self.dataset_name + "/" + self.split + "/"
        labels_path = output_dir + "labels.json"
        output_dir = output_dir + "data"
        with open(labels_path, "r") as f:
            self.data = json.load(f)
        
        images = {img["id"]: img for img in self.data["images"]}
        # Group annotations by image
        img_annotations = {}
        for ann in self.data["annotations"]:
            img_id = ann["image_id"]
            img_annotations.setdefault(img_id, []).append(ann)
        
        # Process each image and save its bboxes
        for img_id, anns in img_annotations.items():
            img_info = images[img_id]
            width = img_info["width"]
            height = img_info["height"]
            file_name = os.path.splitext(img_info["file_name"])[0]
            output_path = os.path.join(output_dir, f"{file_name}.txt")

            # Sort by class (category_id)
            anns_sorted = sorted(anns, key=lambda x: x["category_id"])

            lines = []
            for ann in anns_sorted:
                cat_id = ann["category_id"]
                x, y, w, h = ann["bbox"]


                            # absolute corners
                corners = [
                    (x, y),              # top-left
                    (x + w, y),          # top-right
                    (x + w, y + h),      # bottom-right
                    (x, y + h)           # bottom-left
                ]
                
                # normalize
                normalized = [(cx / width, cy / height) for cx, cy in corners]
                
                # flatten and format like your original dataset
                line = f"{cat_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized])
                
                lines.append(line)

            with open(output_path, "w") as f:
                f.write("\n".join(lines))

        print(f"Saved bounding box files to '{output_dir}'.")





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
