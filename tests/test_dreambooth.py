from mflux import ModelConfig
from mflux.ui.defaults import TRAIN_HEIGHT, TRAIN_WIDTH
from tests.helpers.image_generation_dreambooth_test_helper import ImageGeneratorDreamBoothTestHelper


class TestImageGenerator:
    OUTPUT_IMAGE_FILENAME = "output.png"

    def test_image_generation_dreambooth(self):
        ImageGeneratorDreamBoothTestHelper.assert_matches_reference_image(
            reference_image_path="dreambooth_ref.png",
            output_image_path="dreambooth.png",
            model_config=ModelConfig.FLUX1_DEV,
            steps=20,
            seed=42,
            height=TRAIN_HEIGHT,
            width=TRAIN_WIDTH,
            prompt="photo of sks dog",
            lora_file="/Users/filipstrand/Desktop/0000250_adapters.safetensors",
        )
