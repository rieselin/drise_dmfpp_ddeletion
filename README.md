# On the Black-box Explainability of Object Detection Models for Safe and Trustworthy Industrial Applications

This repository is the official implementation of [On the Black-box Explainability of Object Detection Models for Safe and Trustworthy Industrial Applications](https://www.sciencedirect.com/science/article/pii/S259012302401750X), where algorithms like D-RISE and D-MFPP specifically designed for Object Detection models can be found.
<div align="center">
  <div>
    <img src="imgs_readme/target_object.png" alt="target object" height="250" />
    <img src="imgs_readme/target_object_explanation.png" alt="target object explanation" height="250" />
  </div>
  <p> Figure 1: Target Object and its Explanation generated with D-RISE for (only) 500 masks.</p>
</div>



## Citation
This code is part of our research published in a Q1 journal, [Results in Engineering](https://www.sciencedirect.com/journal/results-in-engineering). If you find it useful, please cite our work
```
@article{AlainAndresXAI2024103498,
  title = {On the Black-box Explainability of Object Detection Models for Safe and Trustworthy Industrial Applications},
  author = {Alain Andres and Aitor Martinez-Seras and Ibai Laña and Javier {Del Ser}},
  journal = {Results in Engineering},
  volume = {24},
  pages = {103498},
  year = {2024},
  issn = {2590-1230},
  doi = {https://doi.org/10.1016/j.rineng.2024.103498},
  url = {https://www.sciencedirect.com/science/article/pii/S259012302401750X},
}
```

## Dependencies

Clone and create a virtual/conda environment. For the first, you can do:

```
python -m venv .venv
source venv/bin/activate
```

Then, install the required dependencies set at requirements.txt:

```
pip install -r requirements.txt
```

## Code structure

```
project
├── metrics                           # Contains code necessary to apply the metrics
├── results                           # Stores the .csv files with quantitative results per image and for each element
├── saliency_maps                     # Place where the attribution maps obtained by each method are saved
├── scripts                           # Contains main logic and notebooks
│   ├── drise_alldetections.py        # Script for D-RISE detections
│   ├── lime_alldetections.py         # Script for LIME detections
│   ├── orise_alldetections.py        # Script for ORISE detections
│   ├── multiple_metrics.py           # Computes quantitative metrics based on provided heatmaps
│   └── jupyter_notebooks             # Folder for Jupyter notebooks (description empty)
├── utils                             # Various utility functions
├── xai                               # Logic for the XAI algorithms
│   ├── base.py                       # Base file for XAI methods
│   ├── drise.py                      # D-RISE implementation
│   ├── orise.py                      # ORISE implementation
│   └── rise.py                       # RISE implementation
└── execute.sh                        # Bash scripts to launch simulations
```

## Example of Use

To use the repository, you can follow these steps:

1. **Execute a Script**: Run any of the scripts in the `scripts` folder (e.g., `drise_alldetections.py`, `lime_alldetections.py`, or `orise_alldetections.py`) to generate the corresponding heatmaps.
2. **Compute Quantitative Results**: Analyze the quantitative results of the selected heatmaps and export them into a `.csv` file for further evaluation.
In addition, the generated heatmaps can be **visualized**. A Jupyter notebook in `scripts/jupyter_notebooks` is provided as an example.


The code is designed to process explanations for multiple images at once. Each XAI method requires the following arguments:

- `--datadir`: Directory containing the test images (e.g., `/path/to/Dataset/images/test/`).
- `--labels_dir`: Directory containing the corresponding labels (e.g., `/path/to/Dataset/labels/test/`).
- `--model_path`: Path to the model weights (e.g., `/path/to/runs/train/model/weights/best.pt`).
- `--height`: Height of the images.
- `--width`: Width of the images.
- `--N`: Number of masks to generate.
- `--p1`: Proportion of occlusion for each mask.
- `--resolution`: Resolution of the mask.

Additional parameters, such as GPU batch size (`gpu_batch`) and devices to use, can also be specified.

### Example Command

Here’s an example of how to run the **D-RISE** method for YOLOv8:

```bash
python3 -m scripts.drise_yolov8_alldetections \
    --datadir /path/to/Dataset/images/test/ \
    --labels_dir /path/to/Dataset/labels/test/ \
    --model_path /path/to/runs/train/model/weights/best.pt \
    --saliency_map_dir saliency_maps/ \
    --height 736 --width 1280 \
    --N 5000 --p1 0.25 --resolution 16 --gpu_batch 50
```

To run **D-MFFP**, set the `mask_type` argument equal to `'mfpp'` (by default, it is set to `'rise'`):
```bash
python3 -m scripts.drise_yolov8_alldetections \
    --datadir /path/to/Dataset/images/test/ \
    --labels_dir /path/to/Dataset/labels/test/ \
    --model_path /path/to/runs/train/model/weights/best.pt \
    --saliency_map_dir saliency_maps/ \
    --height 736 --width 1280 \
    --N 5000 --p1 0.25 --resolution 16 --gpu_batch 50 \
    --mask_type mfpp
```

Following the repository structure, an example command would be:
```bash
python3 -m scripts.drise_yolov8_alldetections  \
    --datadir use_case/ \
    --labels_dir use_case/ \
    --model_path use_case/models/best.pt \
    --saliency_map_dir saliency_maps/ \
    --height 736  --width 1280 \
    --N 500 --p1 0.25 --resolution 16 --gpu_batch 50
```

Once the saliency maps are generated, you can extract evaluation metrics with:
```bash
python3 -m scripts.multiple_metrics  \
    --datadir use_case/ \
    --labels_dir use_case/ \
    --model_path use_case/models/best.pt \
    --saliency_map_dir saliency_maps/ \
    --height 736  --width 1280 \
    --csv_dir "results/metrics.csv" --num_classes 8
```

> 💡 **Note on D-Deletion:**  
> The D-Deletion metric is implemented in `metrics/deletion.py`. Compared to the standard Deletion score, it optionally accepts a `target_bbox` argument and only considers predictions on the modified images that have an IoU greater than 0 with respect to that target.  
> By default, `target_bbox` is set to `None`, so if you want to compute the **D-Deletion** score, make sure to provide it explicitly when calling the evaluation function.


## ACKNOWLEDGEMENTS

This code has been developed with support from the European Commission under the HORIZON-CL4-DIS program. Due to confidentiality agreements and the involvement of stakeholders, the data associated with this project cannot be shared. A single image and a fine-tuned nano model are uploaded as examples on how to use the repository.

<p align="center">
  <img src="imgs_readme/ultimate-logo.jpg" alt="Ultimate Logo" width="300" />
  <img src="imgs_readme/vertical_EU_POS.jpg" alt="Vertical EU POS" width="206" />
</p>

## License Information

This project is licensed under the **Apache 2.0 License** with additional terms for commercial use.

- The software is free to use for **research** and **educational purposes**.
- For **commercial use**, please review the [`LICENSE`](./LICENSE) file and contact the main author, [Alain Andres Fernandez](https://aklein1995.github.io/), for further information.
- **Attribution Requirement**: This statement, alongside the [`LICENSE`](./LICENSE), must be included in any derivative works or distributions.
