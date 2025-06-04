import os
import torch
import argparse
import utils
from data import preprocessing, dataset
from config import *
from models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME, choices=["fasterrcnn", "fasterrcnn_elu"])
    args = parser.parse_args()
    model_name = args.model

    utils.set_seed(RANDOM_SEED)

    # 1. Data preprocessing
    df = preprocessing.load_xview_annotations(ANNOTATIONS_PATH)
    df = preprocessing.filter_invalid_boxes(df)
    df, class_mapping = preprocessing.remap_class_ids(df)
    df = preprocessing.filter_images_with_annotations(df, TRAIN_IMAGES_DIR)
    # For testing: limit to 3 images
    unique_images = df["image_id"].unique()[:3]
    df = df[df["image_id"].isin(unique_images)].copy()

    tiled_df = preprocessing.tile_dataset(df, TRAIN_IMAGES_DIR, N_TILES, TILED_IMAGES_DIR)
    df_train, df_val = preprocessing.split_train_val(tiled_df, test_size=VAL_SPLIT, random_state=RANDOM_SEED)

    # 2. Datasets and loaders
    transform = dataset.get_transforms()
    train_ds = dataset.XViewDataset(df_train, TILED_IMAGES_DIR, transforms=transform)
    val_ds = dataset.XViewDataset(df_val, TILED_IMAGES_DIR, transforms=transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        collate_fn=dataset.collate_fn)
    
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=dataset.collate_fn)

    # 3. Model
    num_classes = len(class_mapping) + 1
    model = get_model(model_name, num_classes)
    model.to(DEVICE)

    # 4. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # 5. Training loop
    train_losses, val_maps = [], []
    map_metric = utils.get_map_metric()
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, targets, _ in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        map_metric.reset()
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                predictions = model(images)
                map_metric.update(predictions, targets)
        epoch_map = map_metric.compute()
        val_maps.append(float(epoch_map["map"]))
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val mAP: {val_maps[-1]:.4f}")

    # 6. Save model
    model_dir = os.path.join(MODEL_SAVE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 6b. Save metrics for evaluation/comparison
    metrics = {"train_losses": train_losses, "val_maps": val_maps}
    utils.save_metrics(metrics, model_name)
    print(f"Metrics saved to {os.path.join(model_dir, 'model_metrics.pkl')}")

    # 7. Plot metrics
    utils.plot_metrics(train_losses, val_maps)

if __name__ == "__main__":
    main()