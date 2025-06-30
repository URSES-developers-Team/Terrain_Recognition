import os
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from config import MODEL_SAVE_DIR, MODEL_NAME, DEVICE, INFERENCE_OUTPUT_DIR
from models import get_model

# YOLO support
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not available. YOLO models will not be supported.")


def load_model(model_path, model_name, num_classes=None):
    """Load model based on model type (FasterRCNN or YOLO)."""
    if "yolo" in model_name.lower():
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package required for YOLO models")
        print(f"Loading YOLO model from: {model_path}")
        return YOLO(model_path)
    else:
        # FasterRCNN models
        print(f"Loading FasterRCNN model from: {model_path}")
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


def print_model_info(model_path, model_name):
    """Print information about the loaded model."""
    try:
        if "yolo" in model_name.lower():
            if not YOLO_AVAILABLE:
                print("Cannot load YOLO model info - ultralytics not available")
                return
            model = YOLO(model_path)
            print(f"Model: {model_path}")
            print(f"Model type: YOLO")
            if hasattr(model.model, 'nc'):
                print(f"Number of classes: {model.model.nc}")
            print(f"Device: {model.device}")
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            final_weight = state_dict["roi_heads.box_predictor.cls_score.weight"]
            num_classes = final_weight.shape[0]
            print(f"Model: {model_path}")
            print(f"Model type: FasterRCNN ({model_name})")
            print(f"Number of classes: {num_classes}")
            print(f"Device: {DEVICE}")
    except Exception as e:
        print(f"Error loading model info: {e}")


def get_enhanced_font():
    """Try to get the best available font for text rendering."""
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)  # macOS
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)  # Linux
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
    return font


def draw_enhanced_detections(image_pil, boxes, labels, scores, score_threshold=0.3, model_type="fasterrcnn"):
    """Enhanced detection visualization with better text rendering."""
    # Create RGBA overlay for transparency
    base_rgba = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    
    # Colors for visualization
    fill_color = (0, 255, 255, 80)  # Cyan with transparency
    outline_color = (0, 0, 0, 255)  # Black outline
    text_color = (255, 255, 255, 255)  # White text
    text_bg_color = (0, 0, 0, 180)  # Semi-transparent black background
    
    # Get enhanced font
    font = get_enhanced_font()
    
    # Draw detections
    for box, lbl, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=2)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
            
            # Prepare label text
            if model_type == "yolo":
                label_text = f"Class {lbl}: {score:.2f}"
            else:
                label_text = f"{lbl}: {score:.2f}"
            
            # Get text size for background
            if font:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = len(label_text) * 8, 16  # Estimate
            
            # Calculate text background position
            text_bg_x1 = x1
            text_bg_y1 = max(0, y1 - text_height - 4)
            text_bg_x2 = min(base_rgba.width, x1 + text_width + 8)
            text_bg_y2 = y1
            
            # Draw text background
            draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], 
                         fill=text_bg_color)
            
            # Draw text
            draw.text((x1 + 4, text_bg_y1 + 2), label_text, fill=text_color, font=font)
    
    # Composite the overlay onto the original image
    result_img = Image.alpha_composite(base_rgba, overlay)
    return result_img.convert("RGB")


def run_inference(model, image_pil, model_name, score_threshold=0.3, conf_threshold=0.25):
    """Run inference based on model type with enhanced visualization."""
    if "yolo" in model_name.lower():
        # YOLO inference
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package required for YOLO models")
        
        img_array = np.array(image_pil)
        results = model(img_array, conf=conf_threshold, verbose=False)
        
        # Convert YOLO results to compatible format
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
        else:
            boxes = np.array([])
            labels = np.array([])
            scores = np.array([])
        
        # Use enhanced visualization
        result_img = draw_enhanced_detections(image_pil, boxes, labels, scores, 
                                            score_threshold, "yolo")
        
    else:
        # FasterRCNN inference
        import torchvision
        transform = torchvision.transforms.ToTensor()
        img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = model(img_tensor)[0]
        
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        
        # Use enhanced visualization
        result_img = draw_enhanced_detections(image_pil, boxes, labels, scores, 
                                            score_threshold, "fasterrcnn")
    
    return result_img


def is_image_file(filename):
    """Check if file is a supported image format."""
    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    return filename.lower().endswith(IMG_EXTS)


def process_image(model, input_path, output_dir, model_name, score_threshold=0.3, conf_threshold=0.25):
    """Process a single image and save the result."""
    try:
        print(f"Processing: {input_path}")
        image = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to open {input_path}: {e}")
        return
    
    # Run inference
    try:
        result_img = run_inference(model, image, model_name, score_threshold, conf_threshold)
    except Exception as e:
        print(f"[ERROR] Inference failed for {input_path}: {e}")
        return
    
    # Save result with appropriate naming
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if "yolo" in model_name.lower():
        output_filename = f"{base_name}_yolo_detections.png"
    else:
        output_filename = f"{base_name}_{model_name}_detections.png"
    
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result_img.save(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"[INFO] Saved: {abs_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced universal object detection inference")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to image file or directory of images")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for annotated images")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                       choices=["fasterrcnn", "fasterrcnn_elu", "fasterrcnn_modified", 
                               "fasterrcnn_ultimate", "yolo", "yolo_enhanced", 
                               "yolo_n", "yolo_s", "yolo_m", "yolo_l", "yolo_x"],
                       help="Model to use")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Custom path to model file (overrides --model)")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                       help="Score threshold for displaying detections")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                       help="Confidence threshold for detection (YOLO only)")
    parser.add_argument("--info", action="store_true",
                       help="Print model information and exit")
    
    args = parser.parse_args()
    
    # Determine model path with proper YOLO naming
    if args.model_path:
        model_path = args.model_path
    else:
        # Map model names to actual directory names
        model_name_mapping = {
            "yolo": "yolov8n",
            "yolo_enhanced": "yolov8n_advanced", 
            "yolo_n": "yolov8n",
            "yolo_s": "yolov8s", 
            "yolo_m": "yolov8m",
            "yolo_l": "yolov8l",
            "yolo_x": "yolov8x"
        }
        
        # Get the actual directory name
        actual_model_name = model_name_mapping.get(args.model, args.model)
        model_dir = os.path.join(MODEL_SAVE_DIR, actual_model_name)
        
        if "yolo" in args.model.lower():
            # YOLO models save in weights/ subdirectory as best.pt
            model_path = os.path.join(model_dir, "weights", "best.pt")
        else:
            model_path = os.path.join(model_dir, "model.pth")  # FasterRCNN uses .pth
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return
    
    # Print model info if requested
    if args.info:
        # Apply same model name mapping for info
        model_name_mapping = {
            "yolo": "yolov8n",
            "yolo_enhanced": "yolov8n_advanced", 
            "yolo_n": "yolov8n",
            "yolo_s": "yolov8s", 
            "yolo_m": "yolov8m",
            "yolo_l": "yolov8l",
            "yolo_x": "yolov8x"
        }
        actual_model_name = model_name_mapping.get(args.model, args.model)
        print_model_info(model_path, actual_model_name)
        return
    
    # Set output directory
    output_dir = args.output if args.output else os.path.join(INFERENCE_OUTPUT_DIR, args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    try:
        model = load_model(model_path, args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Process input
    if os.path.isdir(args.input):
        # Process directory
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                if is_image_file(f)]
        
        if not files:
            print(f"[WARN] No image files found in directory: {args.input}")
            return
        
        print(f"Found {len(files)} image files to process")
        for img_path in files:
            process_image(model, img_path, output_dir, args.model,
                         args.score_threshold, args.conf_threshold)
    
    elif os.path.isfile(args.input) and is_image_file(args.input):
        # Process single file
        process_image(model, args.input, output_dir, args.model,
                     args.score_threshold, args.conf_threshold)
    
    else:
        print(f"[ERROR] Input path is not a valid image or directory: {args.input}")
        return
    
    print(f"Inference complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

