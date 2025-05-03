from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm


def create_captions_dataset():
    # Set up logging to track progress and errors
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load the COCO 2017 dataset
    ds = load_dataset("phiyodr/coco2017", cache_dir="./coco_cache")

    # Only keep the 'captions' column
    logging.info("Extracting captions column...")
    new_ds = ds.map(
        lambda example: {"captions": example["captions"]},
        remove_columns=ds["train"].column_names,
        num_proc=4,  # Use multiple processes to speed up
    )

    # Function to flatten captions into a Dataset
    def create_captions_dataset(split_ds):
        captions = []
        logging.info(f"Processing captions for {split_ds.split} split...")
        for example in tqdm(split_ds, desc=f"Processing {split_ds.split}"):
            captions.extend(example["captions"])
        # Create a Dataset with a proper feature structure
        return Dataset.from_dict({"caption": captions})

    # Create datasets for train and validation captions
    logging.info("Creating train captions dataset...")
    train_captions = create_captions_dataset(new_ds["train"])
    logging.info("Creating validation captions dataset...")
    val_captions = create_captions_dataset(new_ds["validation"])

    # Combine into a DatasetDict
    captions_dict = DatasetDict(
        {"train_captions": train_captions, "val_captions": val_captions}
    )

    # Save to disk with progress logging
    logging.info("Saving captions dataset to disk...")
    captions_dict.save_to_disk("./captions_dataset", num_proc=4)

    logging.info("Processing complete!")


def load_captions_dataset():
    # Load the dataset from disk
    captions_dict = load_dataset("./captions_dataset")
    # Check the loaded dataset
    logging.info("Loaded dataset structure:")
    for split, dataset in captions_dict.items():
        logging.info(f"{split}: {dataset}")
        logging.info(f"First example: {dataset[0]}")
        logging.info(f"Number of examples: {len(dataset)}")
        logging.info(f"Features: {dataset.features}")
    print("Dataset loaded successfully!")
    return captions_dict

# 数据集处理
def preprocess_data(dataset_dict: DatasetDict, batch_size: int,tokenizer):
    def tokenize(examples):
        return tokenizer(
            examples["caption"],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
    num_proc = 8
    dataset_dict = dataset_dict.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc="Tokenizing dataset"
    )
    dataset_dict.set_format("torch", columns=["input_ids", "attention_mask"])
    num_workers = 4
    train_dataloader = DataLoader(
        dataset_dict["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        dataset_dict["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    return train_dataloader, val_dataloader
