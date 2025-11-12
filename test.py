from PIL import Image
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms


def make_detect_grad_friendly(model):
    """
    Patches YOLO Detect layer to allow gradient flow.
    Removes .detach() and no_grad() from the forward pass.
    """
    for module in model.model.modules():
        if module.__class__.__name__ == "Detect":
            print("Patching Detect layer for gradient tracking...")

            old_forward = module.forward

            def new_forward(x):
                # Ensure autograd is enabled for everything
                with torch.enable_grad():
                    preds = old_forward(x)
                    if isinstance(preds, (list, tuple)):
                        preds = [p for p in preds]
                    return preds

            module.forward = new_forward
    return model


if __name__ == "__main__":
    # Load YOLO model
    model = YOLO('yolov8n.pt')  # or your custom model

    # ❗ Do NOT call model.train() here (avoids COCO dataset download)
    model.model.eval()  # inference mode

    # Enable requires_grad manually for all parameters
    for param in model.model.parameters():
        param.requires_grad_(True)

    # Patch detect layer for gradient tracking
    model = make_detect_grad_friendly(model)

    # Dummy input
    #img = torch.randn(1, 3, 640, 640, requires_grad=True)


    img = Image.open('kitti_data/kitti/train/data/000001.png').convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)  # add batch dimension
    img = img.clone().detach().requires_grad_(True)

    # Forward pass with gradients
    with torch.enable_grad():
        preds = model.model(img)
        if isinstance(preds, (list, tuple)):
            loss_like_value = preds[0].mean()
        else:
            loss_like_value = preds.mean()

    # Backward pass
    loss_like_value.backward()
    print("Loss-like value:", loss_like_value.item())

    # Show gradient availability
    grad_count = 0
    for name, param in model.model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            print(f"{name:50s} | grad mean: {param.grad.abs().mean():.9f}")
        else:
            print(f"{name:50s} | grad: None")

    print(f"\nLayers with nonzero gradients: {grad_count}")

    # Plot gradient flow
    layer_names, grad_means = [], []
    for name, param in model.model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_means.append(param.grad.abs().mean().item())

    plt.figure(figsize=(12, 4))
    plt.plot(grad_means)
    plt.title("Gradient Flow Across YOLO Layers")
    plt.xlabel("Layer index")
    plt.ylabel("Mean |grad|")
    plt.tight_layout()
    plt.show()