import mlx.optimizers as optim


class Optimizer:
    @staticmethod
    def setup_optimizer():
        return optim.AdamW(learning_rate=1e-4)
