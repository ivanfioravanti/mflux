from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux import Config, Flux1
from mflux.ui.defaults import TRAIN_HEIGHT, TRAIN_WIDTH


class DreamBoothUtil:
    @staticmethod
    def track_progress(loss: mx.float16, t: int) -> None:
        print(f"Loss: {loss}")

    @staticmethod
    def save_incrementally_and_generate_image(flux: Flux1, t: int) -> None:
        if t % 30 == 0:
            DreamBoothUtil.save_adapter(flux, t)

            image = flux.generate_image(
                seed=42,
                prompt="photo of sks dog",
                config=Config(
                    num_inference_steps=20,
                    width=TRAIN_WIDTH,
                    height=TRAIN_HEIGHT,
                    guidance=4.0,
                ),
            )

            image.save(path=f"/Users/filipstrand/Desktop/test{t:07d}_image.png")

    @staticmethod
    def save_adapter(flux: Flux1, t: int) -> None:
        iteration = t
        out_dir = Path("/Users/filipstrand/Desktop/test")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{iteration:07d}_adapters.safetensors"
        print(f"Saving {str(out_file)}")

        mx.save_safetensors(
            str(out_file),
            dict(tree_flatten(flux.trainable_parameters())),
            metadata={
                "lora_rank": "4",
                "lora_blocks": "1",
            },
        )
