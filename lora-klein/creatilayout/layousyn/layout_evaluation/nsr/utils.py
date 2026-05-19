import json
import numpy as np
from PIL import ImageDraw
from pycocotools.coco import COCO
from tqdm import tqdm

def load_json(fname):
    with open(fname, "r") as file:
        data = json.load(file)
    return data


def get_caption_ids_for_val(nsr_root: str, coco_root: str, out_file: str):
    """
    Save COCO caption_ids for image_ids in NSR-1K validation dataset
    which appear in COCO training dataset.
    """

    files = ["spatial/spatial.val.json", "counting/counting.val.json"]

    image_ids = []
    for file in files:
        data = json.load(open(f"{nsr_root}/{file}", "r"))
        for item in data:
            image_ids.append(item["image_id"])
    image_ids = list(set(image_ids))

    # check caption ids for these image ids in training dataset
    caption_obj = COCO(f"{coco_root}/raw/captions_train2017.json")
    caption_ids = []
    for img_id in image_ids:
        caption_ids.extend(caption_obj.getAnnIds(imgIds=img_id))

    # save caption ids
    print(f"Found {len(caption_ids)} caption ids for {len(image_ids)} images")
    with open(out_file, "w") as f:
        json.dump(caption_ids, f)


def bb_relative_position(boxA, boxB):
    xA_c = (boxA[0] + boxA[2]) / 2
    yA_c = (boxA[1] + boxA[3]) / 2
    xB_c = (boxB[0] + boxB[2]) / 2
    yB_c = (boxB[1] + boxB[3]) / 2
    dist = np.sqrt((xA_c - xB_c) ** 2 + (yA_c - yB_c) ** 2)
    cosAB = (xA_c - xB_c) / dist
    sinAB = (yB_c - yA_c) / dist
    return cosAB, sinAB


def eval_spatial_relation(bbox1, bbox2):
    theta = np.sqrt(2) / 2
    relation = "diagonal"

    if bbox1 == bbox2:
        return relation

    cosine, sine = bb_relative_position(bbox1, bbox2)

    if cosine > theta:
        relation = "right"
    elif sine > theta:
        relation = "top"
    elif cosine < -theta:
        relation = "left"
    elif sine < -theta:
        relation = "bottom"

    return relation


# def create_gif(intermediate_bboxs, labels, output_path, step_size=20, forward=False):
#     # create a gif from the intermediate images
#     frames = []

#     layout_plotter = LayoutPlot(
#         color_map_path=os.path.join(os.path.dirname(__file__), "color_map_coco_grounded.json")
#     )
#     for step_id, bboxs in enumerate(tqdm(intermediate_bboxs, desc="Creating gif")):
#         if step_id % step_size != 0 and step_id != len(intermediate_bboxs) - 1:
#             continue

#         # draw bboxs on image
#         img = plot_bbox_on_img(
#             bboxs,
#             labels,
#             width=256,
#             height=256,
#         )

#         # draw step number
#         draw = ImageDraw.Draw(img)
#         text = f"Step: {step_id}"
#         position = (120, 10)
#         draw.text(position, text, (0, 0, 0))

#         # append to frames
#         frames.append(img)

#     # reverse frames and save gif
#     if not forward:
#         frames = frames[::-1]

#     # save gif
#     frames[0].save(
#         f"{output_path}/sample_layout.gif",
#         save_all=True,
#         append_images=frames[1:],
#         duration=500,
#         loop=0,
#     )
