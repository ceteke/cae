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
    if parsed.fc_layers is not None:
        fc_layers = [int(x) for x in parsed.fc_layers.split('-')]
    else:
        fc_layers = []

    sess = tf.Session()
    swwae = SWWAE(sess,[32,32,3],'autoencode',layers,learning_rate=parsed.learning_rate,lambda_rec=parsed.lambda_rec,
                  lambda_M=parsed.lambda_M,dtype=tf.float32, tensorboard_id=parsed.tensorboard_id, encoder_train=True,
                  fc_ae_layers=fc_layers)

    X, _ = dataset.get_batches(parsed.batch_size)
    X_test, _ = dataset.get_batches(parsed.batch_size, train=False)
    test_steps = len(X_test)

    print("Started training.\nTrain steps: {}".format(len(X)))

    for e in range(parsed.num_epochs):
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

                for test_step in range(test_steps):
                    X_test_step = X_test[test_step]
                    swwae.eval(input=X_test_step)

                print("Train epoch {}:\n\tstep {}\n\tavg. L2 Loss: {}".format(e + 1, step + 1, avg_loss),
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

    if parsed.output_dir is not None:
        swwae.save(path=parsed.output_dir + 'sswae')

    print("Starting test..")

    total_loss = 0.0

    for test_step in range(test_steps):
        X_test_step = X_test[test_step]

        loss = swwae.eval(input=X_test_step)

        total_loss += loss

    print("Test average loss: {}".format(total_loss/test_steps))


if __name__ == "__main__":
   main()