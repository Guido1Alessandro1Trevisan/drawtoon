from logging import Logger
import os
from io import BytesIO
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image
from torch.utils.data import Dataset
from layout_evaluation import Layout
from layousyn.data.utils import crop_layouts

class GriT(Dataset):
    def __init__(
        self,
        path,
        load_image: bool = False,
        crop_augment: bool = False,
        logger: Optional[Logger] = None,
    ):
        self.load_image = load_image
        self.crop_augment = crop_augment

        # List all the Parquet files in the directory
        columns_to_load = [
            "url",
            "caption",
            "width",
            "height",
            "ref_exps",
            "noun_chunks",
        ]
       
        if os.path.isdir(path):
            self.parquet_files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".parquet")
            ]
            self.data = pd.concat(
                [
                    pq.read_table(file, columns=columns_to_load).to_pandas()
                    for file in self.parquet_files
                ],
                ignore_index=True,
            )
        else:
            self.data = pq.read_table(path, columns=columns_to_load).to_pandas()
            
        # filter out samples with less than 3 noun chunks
        self.data = self.data[self.data["noun_chunks"].apply(len) >= 3]

        # Load all data from the Parquet files into a single DataFrame
        if logger:
            logger.info(
                f"Loaded {len(self.data)} samples from Parquet files in {path}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        caption = sample["caption"]
        width = sample["width"]
        height = sample["height"]
        noun_chunks = sample["noun_chunks"]
        
        bboxs, cats = self._get_objects(noun_chunks, caption)
        
        if self.crop_augment:
            bboxs, width, height = crop_layouts(
                bboxs=bboxs,
                width=width,
                height=height,
            )
            
        layout = Layout(
            bboxs=bboxs,
            labels=cats,
            width=width,
            height=height,
        )

        img = None
        if self.load_image:
            url = sample["url"]
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            

        return {
            "caption": caption,
            "layout": layout,
            "image": img,
        }

    def _get_objects(self, nouns, caption):
        bbox = []
        cats = []
        for noun in nouns:
            bbox += [noun[2:6]]
            cats += [caption[int(noun[0]) : int(noun[1])]]
        return bbox, cats

    def get_captions(self, idx):
        sample = self.data.iloc[idx]
        return [sample["caption"]]
