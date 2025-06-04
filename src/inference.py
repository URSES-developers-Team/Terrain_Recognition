import os
import argparse
import torch
from PIL import Image, ImageDraw
from config import MODEL_SAVE_DIR, MODEL_NAME, DEVICE, INFERENCE_OUTPUT_DIR
from models import get_model


def load_model(model_path, model_name, num_classes=None):
    state_dict = torch.load(model_path, map_location=DEVICE)
    # Infer num_classes if not provided
    if num_classes is None:
        final_weight = state_dict["roi_heads.box_predictor.cls_score.weight"]
        num_classes = final_weight.shape[0]
    model = get_model(model_name, num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def run_inference(model, image_pil, score_threshold=0.3):
    import torchvision
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    base_rgba = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    fill_color = (0, 255, 255, 80)
    outline_color = (0, 0, 0, 255)
    text_color = (0, 0, 0, 255)
    font = None
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = None
    for box, lbl, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
            label_text = f"{lbl}: {score:.2f}"
            draw.text((x1, y1), label_text, fill=text_color, font=font)
    result_img = Image.alpha_composite(base_rgba, overlay)
    return result_img.convert("RGB")

def is_image_file(filename):
    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    return filename.lower().endswith(IMG_EXTS)

def process_image(model, input_path, output_dir, score_threshold=0.3):
    try:
        image = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to open {input_path}: {e}")
        return
    result_img = run_inference(model, image, score_threshold=score_threshold)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure all parent dirs exist
    result_img.save(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"[INFO] Saved: {abs_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch object detection and image annotation.")
    parser.add_argument("--input", type=str, required=True, help="Path to an image or directory of images.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save annotated images.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, choices=["fasterrcnn", "fasterrcnn_elu"], help="Model to use.")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Score threshold for displaying detections.")
    args = parser.parse_args()

    output_dir = args.output if args.output else INFERENCE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(MODEL_SAVE_DIR, args.model)
    model_path = os.path.join(model_dir, "model.pth")
    model = load_model(model_path, args.model)

    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if is_image_file(f)]
        if not files:
            print(f"[WARN] No image files found in directory: {args.input}")
            return
        for img_path in files:
            process_image(model, img_path, output_dir, score_threshold=args.score_threshold)
    elif os.path.isfile(args.input) and is_image_file(args.input):
        process_image(model, args.input, output_dir, score_threshold=args.score_threshold)
    else:
        print(f"[ERROR] Input path is not a valid image or directory: {args.input}")

if __name__ == "__main__":
    main()
