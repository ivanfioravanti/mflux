from mflux.dreambooth.finetuning_batch_example import Example


class DreamBoothPreProcessing:
    @staticmethod
    def augment(example: Example) -> list[Example]:
        return [example]
