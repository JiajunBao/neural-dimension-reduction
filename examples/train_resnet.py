import json

from src.models.INSANE.models import InsaneEncoder
from src.models.utils.Trainer import InsaneTrainer


def train():
    with open('models/model_zoo.json', 'r') as istream:
        model_zoo = json.load(istream)
    args = InsaneTrainer.get_solver_arguments()
    encoder = InsaneEncoder.from_scratch(encoder_config=model_zoo[args.config_name],
                                         top_k=args.top_k)
    trainer = InsaneTrainer.from_scratch(model=encoder,
                                         input_dir=args.input_dir,
                                         output_dir=args.output_dir,
                                         learning_rate=args.learning_rate,
                                         n_epoch=args.n_epoch,
                                         per_gpu_batch_size=args.per_gpu_batch_size,
                                         weight_decay=args.weight_decay,
                                         seed=args.seed,
                                         top_k=args.top_k)
    trainer.fit(num_eval_per_epoch=args.num_eval_per_epoch)


if __name__ == '__main__':
    train()



