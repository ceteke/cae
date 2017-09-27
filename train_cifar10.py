from datapy.data.datasets import CIFAR10Dataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_parser
from utils import save_loss, clear_loss
import time

def main():
    clear_loss()
    parser = get_parser()
    parsed = parser.parse_args()

    assert ((parsed.output_dir is None and parsed.save_step is None) or
            (parsed.output_dir is not None and parsed.save_step is not None)), "Save step and output directory must be " \
                                                                               "null at the same time or not null at the same time"

    dataset = CIFAR10Dataset()
    dataset.process()

    layers = parse_layers(parsed.layer_str)

    sess = tf.Session()
    swwae = SWWAE(sess,[32,32,3],'autoencode',layers,parsed.learning_rate,parsed.lambda_rec,parsed.lambda_M,tf.float32)

    X, _ = dataset.get_batches(parsed.batch_size)
    print("Train steps: {}".format(len(X)))

    for e in parsed.num_epochs:
        total_loss = 0.0
        epoch_loss = 0.0
        train_steps = len(X)
        start = time.time()
        for step in range(train_steps):
            X_step = X[step]

            loss, global_step = swwae.train(X_step)

            total_loss += loss
            epoch_loss += loss

            if (step + 1) % parsed.info_step == 0:
                avg_loss = total_loss / parsed.info_step
                save_loss(avg_loss)
                print("Train epoch {}:\n\tstep {}\n\tavg. perplexity: {}".format(e + 1, step + 1, avg_loss),
                      flush=True)
                total_loss = 0.0
                end = time.time()
                hours, rem = divmod(end - start, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Elapsed: {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)), flush=True)
                start = time.time()

            if parsed.save_step is not None:
                if (global_step + 1) % parsed.save_step == 0:
                    swwae.save(path=parsed.output_dir + 'swwae')

        print("Train epoch {}: avg. loss: {}".format(e + 1, epoch_loss / train_steps), flush=True)
        X, _ = dataset.get_batches(parsed.batch_size)


if __name__ == "__main__":
   main()