import argparse
import torch
import utils
from utils import load_metrics
from data import preprocessing, dataset
from config import *
from models import get_model
import os
import matplotlib.pyplot as plt


def plot_model_metrics(model_names):
    for model_name in model_names:
        metrics = load_metrics(model_name)
        train_losses = metrics.get("train_losses", [])
        val_maps = metrics.get("val_maps", [])
        plt.plot(range(1, len(train_losses) + 1), train_losses, label=f"{model_name} Train Loss")
        plt.plot(range(1, len(val_maps) + 1), val_maps, label=f"{model_name} Val mAP")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Validation mAP")
    plt.legend()
    plt.show()


def evaluate_model(model_name, limit=None):
    # 1. Data preprocessing (same as train, but only for val/test)
    df = preprocessing.load_xview_annotations(ANNOTATIONS_PATH)
    df = preprocessing.filter_invalid_boxes(df)
    df, class_mapping = preprocessing.remap_class_ids(df)
    df = preprocessing.filter_images_with_annotations(df, TRAIN_IMAGES_DIR)
    if limit is not None:
        unique_images = df["image_id"].unique()[:limit]
        df = df[df["image_id"].isin(unique_images)].copy()
    tiled_df = preprocessing.tile_dataset(df, TRAIN_IMAGES_DIR, N_TILES, TILED_IMAGES_DIR)
    _, df_val = preprocessing.split_train_val(tiled_df, test_size=VAL_SPLIT, random_state=RANDOM_SEED)

    # 2. Dataset and loader
    transform = dataset.get_transforms()
    val_ds = dataset.XViewDataset(df_val, TILED_IMAGES_DIR, transforms=transform)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=dataset.collate_fn)

    # 3. Model
    num_classes = len(class_mapping) + 1
    model_dir = os.path.join(MODEL_SAVE_DIR, model_name)
    model_path = os.path.join(model_dir, "model.pth")
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 4. Evaluation
    map_metric = utils.get_map_metric()
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            predictions = model(images)
            map_metric.update(predictions, targets)
    results = map_metric.compute()
    print(f"Validation mAP for {model_name}: {results['map']:.4f}")
    return results['map']


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot model metrics.")
    parser.add_argument('--model', type=str, help='Model name to evaluate and plot')
    parser.add_argument('--models', nargs='+', help='List of model names to compare')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of unique images for fast evaluation')
    args = parser.parse_args()

    if args.model:
        evaluate_model(args.model, limit=args.limit)
        plot_model_metrics([args.model])
    elif args.models:
        for model_name in args.models:
            evaluate_model(model_name, limit=args.limit)
        plot_model_metrics(args.models)
    else:
        print("Please provide --model <model> or --models <model1> <model2> ...")


if __name__ == "__main__":
    main()