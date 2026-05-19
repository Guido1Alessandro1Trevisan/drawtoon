import os
from typing import List, Optional

import numpy as np
import torch
from numpy import memmap
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from layout_evaluation import Layout, LayoutType
from layousyn.config import Config
from layousyn.model.t5_google import T5EmbedderGoogle

from .utils import get_padding_mask, pad_to_max_bboxs

NULL_LABEL = ""
NULL_CAPTION = ""
NULL_LOCATION = [-1.0, -1.0, -1.0, -1.0]


@torch.no_grad()
def save_concept_embedding(
    dataset: Dataset,
    sentence_encoder: SentenceTransformer,
    embed_channels: int,
    batch_size: int,
    num_workers: int = 4,
    device: str = "cuda",
    embed_dir: str = "cache/embeds",
    load_only: bool = True,
) -> None:
    """
    Function to save concept embeddings.

    Args:
        dataset (Dataset): The dataset to process.
        embed_channels (int): The number of channels for the concept embeddings.
        max_bboxs (int): The maximum number of bounding boxes per image.
        batch_size (int): The batch size for the DataLoader.
        device (str, optional): The device to run the SentenceTransformer model on. Defaults to 'cuda'.
        save_path (str, optional): The path to save the concept embeddings. Defaults to 'concept_embeddings.pt'.
        flush_interval (int, optional): The interval at which to flush the memmap. Defaults to 1000.
        load_only (bool, optional): Whether to load only. Defaults to True.
    """

    # Define a custom Dataset class
    class ConceptDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            result_dict = self.dataset[idx]
            return result_dict["layout"].labels

    # Define a collate function for the DataLoader
    def collate_fn(batch):
        return batch

    # check if file exists
    if os.path.exists(f"{embed_dir}/concept_embeddings.npy"):
        print(f"Concept embeddings already exist at {embed_dir}. Continuing...")
        return

    # to prevent accidental overwriting
    if load_only:
        raise FileNotFoundError(f"File not found at {embed_dir}")
    else:
        print(f"Creating new concept embeddings at {embed_dir}")

    # Initialize the custom Dataset and DataLoader
    concept_dataset = ConceptDataset(dataset)

    loader = DataLoader(
        concept_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # get unique categories
    unique_concepts = set()
    unique_concepts.update([NULL_LABEL])  # also append '' to the unique categories
    for batch_concepts in tqdm(loader):
        unique_concepts.update(
            [concept for concepts in batch_concepts for concept in concepts]
        )
    unique_concepts = list(unique_concepts)
    print(f"Unique concepts: {len(unique_concepts)}")

    # Start memmap
    concept_embeddings_memmap = memmap(
        f"{embed_dir}/concept_embeddings.npy",
        dtype="float32",
        mode="w+",
        shape=(len(unique_concepts), embed_channels),
    )

    # Encode the categories using the SentenceTransformer model
    concept_embeddings_memmap[:] = sentence_encoder.encode(
        unique_concepts, convert_to_numpy=True, batch_size=batch_size, device=device
    )

    # write the unique categories
    with open(f"{embed_dir}/concept_list.txt", "w") as f:
        for cat in unique_concepts:
            f.write(f"{cat}\n")

    # flush at regular intervals
    concept_embeddings_memmap.flush()

class EmbeddedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        embed_dir: str,
        concept_in_channel: int,
        caption_encoder: T5EmbedderGoogle,
        max_bbox: int,
        layout_type: LayoutType
    ):
        self.dataset = dataset
        self.max_bbox = max_bbox
        self.no_box_label = NULL_LABEL
        self.no_box_location = NULL_LOCATION
        self.caption_encoder = caption_encoder
        self.layout_type = layout_type

        self.concept_embeddings, self.concept_to_memmap_idx = (
            EmbeddedDataset.load_concept_embeddings(embed_dir, concept_in_channel)
        )

    @staticmethod
    def load_concept_embeddings(embed_dir: str, concept_in_channel: int):
        """
        Load concept embeddings from a specified directory.

        Args:
            embed_dir (str): The directory where the concept embeddings are stored.

        Returns:
            tuple: A tuple containing the concept embeddings as a numpy array and a dictionary mapping concept names to their indices.
        """

        # Define the paths to the concept list and embeddings files
        concept_list_file = os.path.join(embed_dir, "concept_list.txt")
        concept_embeddings_file = os.path.join(embed_dir, "concept_embeddings.npy")

        # Initialize an empty list to hold the concept names
        concept_list = []

        # Read the concept names from the concept list file
        with open(concept_list_file, "r") as f:
            for line in f:
                concept_list.append(line.strip("\n"))

        # Load the concept embeddings from the embeddings file
        concept_embeddings = memmap(
            concept_embeddings_file,
            dtype="float32",
            mode="r",
            shape=(len(concept_list), concept_in_channel),
        )
        concept_embeddings = np.array(concept_embeddings)

        # Convert the list of concept names to a dictionary mapping concept names to indices
        concept_list = EmbeddedDataset.map_liststr_to_dict(concept_list)

        return concept_embeddings, concept_list

    @staticmethod
    def store_embeddings(dataset, config: Config, label_encoder, device):
        # ensure embed_dir exists
        os.makedirs(config.embed_dir, exist_ok=True)

        # save concept embeddings
        save_concept_embedding(
            dataset,
            label_encoder,
            embed_channels=config.concept_in_channel,
            batch_size=config.concept_embedder_batch_size,
            num_workers=config.num_workers,
            device=device,
            embed_dir=config.embed_dir,
            load_only=not config.save_embed,
        )

    @staticmethod
    def map_liststr_to_dict(list: List[str]):
        return {list[i]: i for i in range(len(list))}

    def _get_cat_embedding(self, cats):
        cat_embedding = [
            self.concept_embeddings[self.concept_to_memmap_idx[cat]] for cat in cats
        ]
        return torch.tensor(np.array(cat_embedding, dtype=np.float32))

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        result_dict = self.dataset[idx]
        caption = result_dict["caption"]
        layout: Layout = result_dict["layout"]
        layout = layout.to(self.layout_type)

        # pad to max_bbox
        bboxs = layout.bboxs
        cats = layout.labels
        len_cats_unpadded = len(cats)
        if self.max_bbox:
            bboxs, cats = pad_to_max_bboxs(
                bboxs, cats, self.max_bbox, self.no_box_location, self.no_box_label
            )
        bboxs = torch.tensor(bboxs, dtype=torch.float32)  # (max_bboxs, 4)
        bboxs_padding_mask = torch.tensor(get_padding_mask(cats, len_cats_unpadded))

        # get cat embeddings
        cats_encoded = self._get_cat_embedding(cats)

        # get inout ids
        _, caption_input_ids, caption_attention_mask = self.caption_encoder.get_tokens(
            [caption]
        )

        # get aspect ratio
        aspect_ratio = layout.width / layout.height

        return (
            bboxs,
            bboxs_padding_mask,
            cats_encoded,
            caption_input_ids.squeeze(0),
            caption_attention_mask.squeeze(0),
            aspect_ratio,
        )

    @staticmethod
    def get_null_caption_embedding(caption_encoder: T5EmbedderGoogle):
        _, caption_embedding, caption_attention_mask = caption_encoder.get_text_embeddings([NULL_CAPTION])
        return caption_embedding[0], caption_attention_mask[0] == 0