import pandas as pd
import json
import os
from PIL import Image
import math

def load_xview_annotations(geojson_path: str) -> pd.DataFrame:
    """
    Reads xView GeoJSON file and parses bounding box annotations into a DataFrame.
    Returns columns: ["image_id", "x_min", "y_min", "x_max", "y_max", "class_id"].
    """
    with open(geojson_path, "r") as f:
        data = json.load(f)

    rows = []
    for feature in data["features"]:
        props = feature["properties"]
        image_id = props["image_id"]
        type_id = props["type_id"]

        imcoords_str = props.get("bounds_imcoords", "")
        if imcoords_str:
            x1, y1, x2, y2 = map(float, imcoords_str.split(","))
            rows.append({
                "image_id": image_id,
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
                "class_id": type_id
            })

    return pd.DataFrame(rows)

def filter_invalid_boxes(df):
    """
    Removes bounding boxes with non-pistive width or height.
    """
    df = df.copy()
    df["width"] = df["x_max"] - df["x_min"]
    df["height"] = df["y_max"] - df["y_min"]
    df = df[(df["width"] > 0) & (df["height"] > 0)]
    df = df.drop(columns=["width", "height"])
    return df

def remap_class_ids(df):
    """
    Remaps class IDs to [1..62], 0 is reserved for bg.
    Return: new DataFrame and the mapping dictionary.
    """
    df = df.copy()
    unique_classes = sorted(df["class_id"].unique())
    class_mapping = {old_id: idx + 1 for idx, old_id in enumerate(unique_classes)}
    df["class_id"] = df["class_id"].map(class_mapping)
    return df, class_mapping

def filter_images_with_annotations(df, images_dir):
    """
    Keeps only annotations for images that exist in the images directory.
    """
    valid_files = set(os.listdir(images_dir))
    return df[df["image_id"].isin(valid_files)].copy()

def tile_image(image_path, bboxes_df, n_tiles, output_dir):
    """
    Splits a single image into about `n_tiles` smaller tiles without cutting bounding boxes.
    Only bounding boxes fully contained in a tile are kept.
    Returns a DataFrame of tiled annotations.
    """
    image = Image.open(image_path).convert("RGB")
    img_name = os.path.basename(image_path)
    W, H = image.size

    Nx = int(math.floor(math.sqrt(n_tiles)))
    if Nx < 1:
        Nx = 1
    Ny = int(math.ceil(n_tiles / Nx))

    tile_width = W // Nx
    tile_height = W // Ny

    tiled_annotations = []

    for ix in range(Nx):
        for iy in range(Ny):
            x_start = ix * tile_width
            x_end = (ix + 1) * tile_width if (ix + 1) < Nx else W
            y_start = iy * tile_height
            y_end = (iy + 1) * tile_height if (iy + 1) < Ny else H

            inside_tile = bboxes_df[
                (bboxes_df["x_min"] >= x_start) &
                (bboxes_df["x_max"] <= x_end) &
                (bboxes_df["y_min"] >= y_start) &
                (bboxes_df["y_max"] <= y_end)
                ]
            
            if inside_tile.empty:
                continue

            tile = image.crop((x_start, y_start, x_end, y_end))
            tile_name = f"{os.path.splitext(img_name)[0]}_{x_start}_{y_start}.jpg"
            tile_path = os.path.join(output_dir, tile_name)
            tile.save(tile_path)

            for _, row in inside_tile.iterrows():
                new_xmin = row["x_min"] - x_start
                new_xmax = row["x_max"] - x_start
                new_ymin = row["y_min"] - y_start
                new_ymax = row["y_max"] - y_start

                tiled_annotations.append({
                    "image_id": tile_name,
                    "x_min": new_xmin,
                    "x_max": new_xmax,
                    "y_min": new_ymin,
                    "y_max": new_ymax,
                    "class_id": row["class_id"]
                })
    return pd.DataFrame(tiled_annotations)

def tile_dataset(df, images_dir, n_tiles, output_dir):
    """
    Tiles all imagees in the dataset and returns concatenated DF of all tiled annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_tiled_rows = []
    grouped = df.groupby("image_id")
    for image_id, group_df in grouped:
        img_path = os.path.join(images_dir, image_id)
        if not os.path.exists(img_path):
            continue
        tiled_df = tile_image(
            image_path=img_path,
            bboxes_df=group_df,
            n_tiles=n_tiles,
            output_dir=output_dir
        )
        all_tiled_rows.append(tiled_df)
    if all_tiled_rows:
        return pd.concat(all_tiled_rows, ignore_index=True)
    return pd.DataFrame(columns=["image_id","x_min", "y_min", "x_max", "y_max", "class_id"])

def split_train_val(df, test_size=0.2, random_state=42):
    """
    Split the tiled dataset into train and validation sets by image_id.
    Return: train_df, val_df
    """
    from sklearn.model_selection import train_test_split
    unique_tile_images = df["image_id"].unique()
    train_ids, val_ids = train_test_split(unique_tile_images, test_size=test_size, random_state=random_state)
    df_train = df[df["image_id"].isin(train_ids)].copy()
    df_val = df[df["image_id"].isin(val_ids)].copy()
    return df_train, df_val