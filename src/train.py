import os
import torch
import argparse
import utils
from data import preprocessing, dataset
from config import *
from models import get_model
import torch.optim as optim
from datetime import datetime


def get_scheduler(optimizer, scheduler_type, total_epochs, step_size, gamma):
    """
    Returns a learning rate scheduler based on config.
    Args:
        optimizer: The optimizer instance.
        scheduler_type: 'cosine', 'cyclic', or 'step'.
        total_epochs: Total number of epochs (for cosine/cyclic).
        step_size: Step size for StepLR.
        gamma: Decay factor for StepLR.
    """
    if scheduler_type == 'cosine':
        # Cosine Annealing with warm restarts
        print(f"📈 Using CosineAnnealingLR: T_max={total_epochs}")
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    elif scheduler_type == 'cyclic':
        # CyclicLR: cycles between base_lr and max_lr
        print(f"📈 Using CyclicLR: base_lr=1e-5, max_lr=5e-4, step_size_up={total_epochs//4}")
        return optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=1e-5, max_lr=5e-4, step_size_up=total_epochs//4, mode='triangular2'
        )
    elif scheduler_type == 'step':
        print(f"📈 Using StepLR: step_size={step_size}, gamma={gamma}")
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME, 
                       choices=["base", "enhanced", "ultimate", "apex"])
    args = parser.parse_args()
    model_name = args.model

    utils.set_seed(RANDOM_SEED)

    # 1. Data preprocessing
    df = preprocessing.load_xview_annotations(ANNOTATIONS_PATH)
    df = preprocessing.filter_invalid_boxes(df)
    df, class_mapping = preprocessing.remap_class_ids(df)
    df = preprocessing.filter_images_with_annotations(df, TRAIN_IMAGES_DIR)
    # For testing: limit to 3 images
    # unique_images = df["image_id"].unique()[:3]
    # df = df[df["image_id"].isin(unique_images)].copy()

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
    
    # For ultimate/apex models, we need to pass class_counts
    if model_name in ["ultimate", "apex"]:
        # Compute class counts from the preprocessed data
        class_counts = {}
        for class_id in tiled_df["class_id"].unique():
            class_counts[class_id] = len(tiled_df[tiled_df["class_id"] == class_id])
        
        print(f"📊 Class distribution computed:")
        print(f"   Total classes: {len(class_counts)}")
        print(f"   Most common: {max(class_counts.values())} samples")
        print(f"   Least common: {min(class_counts.values())} samples")
        print(f"   Imbalance ratio: {max(class_counts.values())/min(class_counts.values()):.1f}:1")
        
        if model_name == "ultimate":
            # Import ultimate model directly to avoid preprocessing duplication
            from models.faster_rcnn.ultimate import UltimateFasterRCNN
            model = UltimateFasterRCNN(num_classes, class_counts, DEVICE.type)
        elif model_name == "apex":
            # Import apex model (simplified version)
            from models.faster_rcnn.apex_simplified import ApexFasterRCNN
            model = ApexFasterRCNN(num_classes, class_counts, DEVICE.type, enable_amp=True)
    else:
        model = get_model(model_name, num_classes)
    
    model.to(DEVICE)

    # 4. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Use model-specific learning rate recommendations for ultimate/apex models
    if model_name in ["ultimate", "apex"]:
        recommendations = model.get_training_recommendations()
        lr = recommendations['learning_rate']
        wd = recommendations['weight_decay']
        print(f"📚 Using {model_name.title()} model recommendations: LR={lr}, WD={wd}")
    else:
        lr = LEARNING_RATE
        wd = WEIGHT_DECAY
        
    optimizer = torch.optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=wd)
    
    # Add learning rate scheduler for ultimate/apex models
    scheduler = None
    if model_name in ["ultimate", "apex"]:
        recommendations = model.get_training_recommendations()
        scheduler = get_scheduler(
            optimizer,
            scheduler_type=SCHEDULER_TYPE,
            total_epochs=SCHEDULER_TOTAL_EPOCHS,
            step_size=SCHEDULER_STEP_SIZE,
            gamma=SCHEDULER_GAMMA
        )
        print(f"📈 Scheduler type: {SCHEDULER_TYPE}")
        # For Apex model, also print AMP status
        if model_name == "apex":
            print(f"⚡ Mixed-Precision Training: {'Enabled' if model.enable_amp else 'Disabled'}")

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
            
            # Use enhanced training for Apex model
            if model_name == "apex":
                loss_dict, total_loss = model.train_step(images, targets, optimizer)
            else:
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())
                
                # Check for NaN/Inf in loss before backward pass
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: NaN/Inf loss detected, skipping batch. Loss: {total_loss}")
                    continue
                    
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val mAP: {val_maps[-1]:.4f} | {timestamp}")
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"   Learning Rate: {current_lr:.6f}")

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
