import os
import argparse
import utils
from data import preprocessing, yolo_dataset
from config import *
from models.yolo.yolo import train_yolo_native


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on xView dataset")
    
    # Usage examples:
    # Regular YOLO:     python train_yolo.py --model-size m
    # Enhanced YOLO:    python train_yolo.py --model-size m --enhanced
    # (Enhanced is equivalent to yolo_enhanced_m from the model factory)
    
    parser.add_argument("--model-size", type=str, default="n", 
                       choices=["n", "s", "m", "l", "x"],
                       help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of images for testing")
    parser.add_argument("--enhanced", action="store_true",
                       help="Use Enhanced YOLO with satellite imagery optimizations (equivalent to yolo_enhanced_<size>)")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                       help="Weight decay for regularization")
    args = parser.parse_args()

    utils.set_seed(RANDOM_SEED)

    print(f"Training YOLOv8{args.model_size} on xView satellite imagery dataset")
    print(f"Device: {DEVICE}")

    # 1. Data preprocessing (same as original)
    print("Loading and preprocessing data...")
    df = preprocessing.load_xview_annotations(ANNOTATIONS_PATH)
    df = preprocessing.filter_invalid_boxes(df)
    df, class_mapping = preprocessing.remap_class_ids(df)
    df = preprocessing.filter_images_with_annotations(df, TRAIN_IMAGES_DIR)
    # For testing: limit to 3 images
    unique_images = df["image_id"].unique()[:3]
    df = df[df["image_id"].isin(unique_images)].copy()

    # For testing with limited data
    if args.limit:
        unique_images = df["image_id"].unique()[:args.limit]
        df = df[df["image_id"].isin(unique_images)].copy()
        print(f"Limited dataset to {args.limit} images for testing")

    # 2. Tile the images
    print("Tiling images...")
    tiled_df = preprocessing.tile_dataset(df, TRAIN_IMAGES_DIR, N_TILES, TILED_IMAGES_DIR)
    df_train, df_val = preprocessing.split_train_val(tiled_df, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    
    print(f"Training set: {len(df_train)} annotations, {df_train['image_id'].nunique()} images")
    print(f"Validation set: {len(df_val)} annotations, {df_val['image_id'].nunique()} images")

    # 3. Prepare YOLO dataset format
    print("Converting to YOLO format...")
    yolo_dir, yaml_path = yolo_dataset.prepare_yolo_dataset(df_train, df_val, class_mapping)

    # 4. Train the model using native YOLO training
    training_mode = "Enhanced" if args.enhanced else "Standard"
    print(f"Starting native YOLO training with {training_mode} parameters...")
    
    # Prepare training arguments
    training_kwargs = {
        'workers': NUM_WORKERS,
        'project': MODEL_SAVE_DIR,
        'name': f"yolov8{args.model_size}{'_enhanced' if args.enhanced else ''}",
        'exist_ok': True
    }
    
    # Add enhanced configurations for satellite imagery
    if args.enhanced:
        print("🚀 Using enhanced training configuration for satellite imagery:")
        training_kwargs.update({
            # Enhanced data augmentation for satellite imagery
            'hsv_h': 0.010,      # Reduced hue variation (satellite imagery has consistent lighting)
            'hsv_s': 0.5,        # Moderate saturation variation
            'hsv_v': 0.3,        # Moderate value variation
            'degrees': 0.2,      # Reduced rotation (satellite imagery is typically north-up)
            'translate': 0.05,   # Reduced translation (precise positioning important)
            'scale': 0.3,        # Reduced scaling (object sizes are important)
            'shear': 0.0,        # No shear (satellite imagery is orthogonal)
            'perspective': 0.0,  # No perspective (satellite imagery is orthogonal)
            'flipud': 0.0,       # No vertical flip (orientation matters)
            'fliplr': 0.3,       # Some horizontal flip is acceptable
            'mosaic': 0.8,       # Reduced mosaic (preserve spatial relationships)
            'mixup': 0.05,       # Reduced mixup (preserve object integrity)
            'copy_paste': 0.05,  # Reduced copy-paste (maintain spatial context)
            
            # Enhanced loss configuration
            'cls': 0.3,          # Reduced classification loss (focus on detection)
            'box': 10.0,         # Increased box regression loss (precise localization)
            'dfl': 2.0,          # Increased distribution focal loss (better boundaries)
            
            # Enhanced training schedule
            'warmup_epochs': 5.0,    # More warmup for stable training
            'warmup_momentum': 0.5,  # Conservative warmup momentum
            'cos_lr': True,          # Cosine learning rate schedule
            'close_mosaic': 15,      # Close mosaic augmentation earlier
            
            # Enhanced regularization
            'dropout': 0.2,          # Add dropout for regularization
            'weight_decay': args.weight_decay * 1.5,  # Increased weight decay
            
            # Satellite-specific optimizations
            'optimizer': 'AdamW',    # Better optimizer for complex scenes
            'lr0': args.learning_rate * 0.8,  # Slightly lower initial LR for stability
            'lrf': 0.001,           # Lower final learning rate
        })
        print("  ✅ Enhanced augmentation for satellite imagery")
        print("  ✅ Optimized loss weights for object detection")
        print("  ✅ Enhanced learning rate scheduling")
        print("  ✅ Improved regularization")
    else:
        print("📋 Using STANDARD training configuration:")
        training_kwargs.update({
            'lr0': args.learning_rate,
            'weight_decay': args.weight_decay,
        })
    
    results = train_yolo_native(
        model_path_or_size=args.model_size,
        data_yaml_path=yaml_path,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        **training_kwargs
    )

    # 5. Save final model in our format
    model_name = f"yolov8{args.model_size}{'_enhanced' if args.enhanced else ''}"
    model_dir = os.path.join(MODEL_SAVE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # The trained model is automatically saved by YOLO training
    print(f"Training completed!")
    print(f"Models saved to: {os.path.join(MODEL_SAVE_DIR, f'yolov8{args.model_size}')}")
    
    # Print final metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"Final validation mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Final validation mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    else:
        print("Training metrics available in the training output above.")


if __name__ == "__main__":
    main()
