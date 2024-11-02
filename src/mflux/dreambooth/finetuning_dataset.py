import json
from pathlib import Path

import mlx.core as mx
import PIL.Image
from mlx import nn
from tqdm import tqdm

from mflux import Flux1, ImageUtil
from mflux.dreambooth.dreambooth_preprocessing import DreamBoothPreProcessing
from mflux.dreambooth.finetuning_batch_example import Example
from mflux.dreambooth.finetuning_dataset_iterator import DatasetIterator
from mflux.post_processing.array_util import ArrayUtil
from mflux.ui.defaults import TRAIN_HEIGHT, TRAIN_WIDTH


class FineTuningDataset:
    def __init__(self, raw_data: list[dict[str, str]], root_dir: Path):
        self.raw_data = raw_data
        self.root_dir = root_dir
        self.prepared_examples = None

    @staticmethod
    def load_from_disk(path: str | Path) -> "FineTuningDataset":
        path = Path(path)
        index_file = path / "index.json"

        with open(index_file, "r") as f:
            data = json.load(f)

        return FineTuningDataset(raw_data=data, root_dir=path)

    def prepare_dataset(self, flux: Flux1) -> None:
        # Encode the original examples (image and text)
        examples = self._create_examples(flux, self.raw_data)

        # Expend the original dataset to get more training data with variations
        augmented_examples = []
        for example in examples:
            [augmented_examples.append(variation) for variation in DreamBoothPreProcessing.augment(example)]

        # Dataset is now prepared
        self.prepared_examples = augmented_examples

    def get_iterator(self, batch_size: int = 1) -> iter:
        return iter(DatasetIterator(self, batch_size=batch_size))

    def _create_examples(self, flux: Flux1, raw_data: list[dict[str, str]]) -> list[Example]:
        examples = []
        for i, entry in enumerate(tqdm(raw_data, desc="Encoding original dataset")):
            # Encode the image
            image_path = self.root_dir / entry["image"]
            encoded_image = FineTuningDataset._encode_image(flux.vae, image_path)

            # Encode the prompt
            prompt = entry["prompt"]
            prompt_embeds = flux.t5_text_encoder.forward(flux.t5_tokenizer.tokenize(prompt))
            pooled_prompt_embeds = flux.clip_text_encoder.forward(flux.clip_tokenizer.tokenize(prompt))

            # Create the example object
            example = Example(
                prompt=prompt,
                image_path=image_path,
                encoded_image=encoded_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )
            examples.append(example)

            # Evaluate to enable progress tracking
            mx.eval(encoded_image)
            mx.eval(prompt_embeds)
            mx.eval(pooled_prompt_embeds)

        return examples

    @staticmethod
    def _encode_image(vae: nn.Module, image_path: Path) -> mx.array:
        image = PIL.Image.open(image_path)
        scaled_user_image = ImageUtil.scale_to_dimensions(image, target_width=TRAIN_WIDTH, target_height=TRAIN_HEIGHT)
        encoded = vae.encode(ImageUtil.to_array(scaled_user_image))
        latents = ArrayUtil.pack_latents(encoded, width=TRAIN_WIDTH, height=TRAIN_HEIGHT)
        return latents
