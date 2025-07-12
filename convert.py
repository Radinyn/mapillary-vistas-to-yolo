from os import listdir
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import cv2


def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def list_labels():
    vistas_config_path = Path(load_config()["vistas_path"]) / "config_v2.0.json"

    with open(vistas_config_path, "r") as file:
        vistas_config = json.load(file)
        labels = vistas_config["labels"]

        print("labels:")
        for i, label in enumerate(labels):
            print(f"\t{label['name']}: {i}")


def create_relabel_map():
    config = load_config()
    label_to_number = config["labels"]
    number_to_relabel = config["relabels"]

    # old_label:(new_label_index, new_label)
    return {
        k: (v, number_to_relabel[v]) for k, v in label_to_number.items() if v != "x"
    }


def polygon_to_normalized_xywh(polygon, img_width, img_heigth):
    # polygon: [[x1, y1], [x2, y2]...]

    xs, ys = zip(*polygon)

    x_max = max(xs)
    x_min = min(xs)
    y_max = max(ys)
    y_min = min(ys)

    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_heigth
    x_center = ((x_max + x_min) / 2) / img_width
    y_center = ((y_max + y_min) / 2) / img_heigth

    return (x_center, y_center, w, h)


def convert_dataset(dir):
    config = load_config()

    vistas_path = Path(config["vistas_path"])
    polygons_path = vistas_path / dir / "v2.0" / "polygons"

    # according to the Ultralytics YOLO specification
    images_path = vistas_path / dir / "images"
    yolo_path = vistas_path / dir / "v2.0" / "yolo"

    yolo_path.mkdir(parents=True, exist_ok=True)

    polygons_filenames = listdir(polygons_path)

    relabel_map = create_relabel_map()

    for filename in tqdm(
        polygons_filenames, desc=f"Conversion progress ({dir})", position=0, leave=True
    ):
        with open(polygons_path / filename, "r") as file:
            objects = json.load(file)["objects"]

            # strip .json, add .jpg
            filename_stem = filename[:-5]

            img_filename = filename_stem + ".jpg"
            txt_filename = filename_stem + ".txt"

            img_heigth, img_width, *_ = cv2.imread(images_path / img_filename).shape

            with open(yolo_path / txt_filename, "w") as out_file:
                result = ""
                for object in objects:
                    if object["label"] not in relabel_map:
                        continue

                    relabel_number = relabel_map[object["label"]][0]
                    x, y, w, h = polygon_to_normalized_xywh(
                        object["polygon"], img_width, img_heigth
                    )

                    result += f"{relabel_number} {x:.7f} {y:.7f} {w:.7f} {h:.7f}\n"
                out_file.write(result)


def generate_yolo_cfg():
    relabel_map = create_relabel_map()
    with open("vistas.yaml", "w") as file:
        file.write("path: ###\n")
        file.write("train: ###\n")
        file.write("val: ###\n")
        file.write("test:\n")
        file.write("\n")
        file.write("names:\n")

        pairs = {
            (index, new_label) for _old_label, (index, new_label) in relabel_map.items()
        }
        pairs = sorted(list(pairs))

        for index, new_label in pairs:
            file.write(f"  {index}: {new_label}\n")


if __name__ == "__main__":
    convert_dataset("training")
    convert_dataset("validation")
    generate_yolo_cfg()
    print("Generated vistas.yaml config.")
    print("--- DONE ---")
