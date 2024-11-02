from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from PIL import Image

from mflux import Config, Flux1, ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.weights.weight_util import WeightUtil
from tests.helpers.image_generation_test_helper import ImageGeneratorTestHelper


class ImageGeneratorDreamBoothTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_config: ModelConfig,
        prompt: str,
        steps: int,
        seed: int,
        lora_file: str,
        height: int = None,
        width: int = None,
        init_image_path: str | None = None,
        init_image_strength: float | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorTestHelper.resolve_path(output_image_path)
        lora_paths = [str(ImageGeneratorTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        try:
            # Set up the model
            runtime_config = RuntimeConfig(
                model_config=model_config,
                config=Config(
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    guidance=4.0,
                ),
            )
            flux = Flux1(model_config=runtime_config.model_config, quantize=8)
            flux.set_lora_layer()
            lora = ImageGeneratorDreamBoothTestHelper.load(lora_file)
            flux.update(lora)

            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    guidance=4.0,
                ),
            )

            image.save(path=output_image_path)

            # then
            np.testing.assert_array_equal(
                np.array(Image.open(output_image_path)),
                np.array(Image.open(reference_image_path)),
                err_msg=f"Generated image doesn't match reference image. Check {output_image_path} vs {reference_image_path}",
            )

        finally:
            # cleanup
            print("test")
            # if os.path.exists(output_image_path):
            #     os.remove(output_image_path)

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent / "resources" / path

    @staticmethod
    def load(file1: str):
        weight = list(mx.load(str(file1)).items())
        weights = [weight]

        # Huggingface weights needs to be reshaped
        weights = WeightUtil.flatten(weights)
        unflatten = tree_unflatten(weights)
        return unflatten
