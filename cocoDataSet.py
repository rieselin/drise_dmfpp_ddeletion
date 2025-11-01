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
    def save_bboxes_to_file(self , overwrite=False):
        # save bboxes in class_id x_min y_min x_max y_max format (as text file imagename.txt)
        output_dir = self.dataset_dir +"/"+ self.dataset_name + "/" + self.split + "/"
        labels_path = output_dir + "labels.json"
        output_dir = output_dir + "data"

        # check if labelfiles already exist
        if not overwrite:
            existing_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
            if len(existing_files) >= self.max_samples:
                print(f"Label files already exist in '{output_dir}'. Skipping save.")
                return

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



