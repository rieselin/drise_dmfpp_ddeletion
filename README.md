# On the Black-box Explainability of Object Detection Models for Safe and Trustworthy Industrial Applications

## Citation

```
@misc{andres2024blackboxexplainabilityobjectdetection,
      title={On the Black-box Explainability of Object Detection Models for Safe and Trustworthy Industrial Applications}, 
      author={Alain Andres and Aitor Martinez-Seras and Ibai Laña and Javier Del Ser},
      year={2024},
      eprint={2411.00818},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.00818}, 
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
2. **Visualize Results**: Open the generated results in the Jupyter notebooks found within the `jupyter_notebooks` subfolder in `scripts`.
3. **Compute Quantitative Results**: Analyze the quantitative results of the selected heatmaps and export them into a `.csv` file for further evaluation.

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

Here’s an example of how to run the D-RISE method for YOLOv8 with specified parameters:

```bash
python3 -m scripts.drise_yolov8_alldetections \
    --datadir /path/to/Dataset/images/test/ \
    --labels_dir /path/to/Dataset/labels/test/ \
    --model_path /path/to/runs/train/model/weights/best.pt \
    --saliency_map_dir saliency_maps/ \
    --height 736 --width 1280 \
    --N 5000 --p1 0.25 --resolution 16 --gpu_batch 50
```

```bash
python3 -m scripts.drise_yolov8_alldetections  \
    --datadir use_case/ \
    --labels_dir use_case/ \
    --model_path use_case/models/best.pt \
    --saliency_map_dir saliency_maps/ \
    --height 736  --width 1280 \
    --N 500 --p1 0.25 --resolution 16 --gpu_batch 50
```

Then, extract metrics with:

```bash
python3 -m scripts.multiple_metrics  \
    --datadir use_case/ \
    --labels_dir use_case/ \
    --model_path use_case/models/best.pt \
    --saliency_map_dir saliency_maps/ \
    --height 736  --width 1280 \
    --csv_dir "results/metrics.csv" --num_classes 8
```

## ACKNOWLEDGEMENTS

This code has been developed with support from the European Commission under the HORIZON-CL4-DIS program. Due to confidentiality agreements and the involvement of stakeholders, the data associated with this project cannot be shared. A single image and a fine-tuned nano model are uploaded as examples on how to use the repository.


### License Information

This project is licensed under the Apache 2.0 License with additional terms for commercial use. The software is free for research and educational purposes. For any commercial use, please review the `LICENSE` file and contact the main author (https://aklein1995.github.io/) for further information.
