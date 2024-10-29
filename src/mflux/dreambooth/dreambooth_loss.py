import mlx.core as mx

from mflux import Config, Flux1
from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth.finetuning_batch_example import Batch
from mflux.dreambooth.finetuning_dataset import Example
from mflux.latent_creator.latent_creator import LatentCreator


class DreamBoothLoss:
    @staticmethod
    def compute_loss(flux: Flux1, config: RuntimeConfig, batch: Batch) -> mx.float16:
        losses = [DreamBoothLoss._single_example_loss(flux, config, example) for example in batch.examples]
        return mx.mean(mx.array(losses))

    @staticmethod
    def _single_example_loss(flux: Flux1, config: RuntimeConfig, example: Example) -> mx.float16:
        # Draw a random timestep t from [0, num_inference_steps]
        t = int(
            mx.random.randint(
                low=0,
                high=config.num_inference_steps,
                shape=[],
            )
        )  # fmt: off

        # Get the clean image latent
        clean_image = example.clean_latents

        # Generate pure noise
        pure_noise = mx.random.normal(
            shape=clean_image.shape,
            dtype=Config.precision,
        )  # fmt: off

        # By linear interpolation between the clean image and pure noise, construct a latent at time t
        latents_t = LatentCreator.add_noise_by_interpolation(
            clean=clean_image,
            noise=pure_noise,
            sigma=config.sigmas[t]
        )  # fmt: off

        # Predict the noise from timestep t
        predicted_noise = flux.transformer.predict(
            t=t,
            prompt_embeds=example.prompt_embeds,
            pooled_prompt_embeds=example.pooled_prompt_embeds,
            hidden_states=latents_t,
            config=config,
        )

        # Construct the loss
        return ((clean_image + predicted_noise) - pure_noise).square().mean()
