from datapy.data.datasets import CIFAR10Dataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_class_parser
from utils import save_loss, clear_loss, accuracy
import time

def main():
    clear_loss()
    parser = get_class_parser()
    parsed = parser.parse_args()

    assert ((parsed.output_dir is None and parsed.save_step is None) or
            (parsed.output_dir is not None and parsed.save_step is not None)), "Save step and output directory must be " \
                                                                               "null at the same time or not null at the same time"

    dataset = CIFAR10Dataset()
    dataset.process()

    encoder_train = True if parsed.encoder_train==1 else False

    layers = parse_layers(parsed.layer_str)
    fc_layers = parsed.fc_layers.split('-')

    sess = tf.Session()
    swwae = SWWAE(sess,[32,32,3],'classification',layers,fc_layers=fc_layers,learning_rate=parsed.learning_rate,
                  tensorboard_id=parsed.tensorboard_id, num_classes=10, encoder_train=encoder_train)
    swwae.restore(parsed.load_dir)

    X, y = dataset.get_batches(parsed.batch_size)
    X_test, y_test = dataset.get_batches(parsed.batch_size, train=False)
    print("Started training.\nTrain steps: {}".format(len(X)))

    for e in range(parsed.num_epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        epoch_loss = 0.0
        epoch_acc = 0.0
        train_steps = len(X)
        start = time.time()
        for step in range(train_steps):
            X_step = X[step]
            y_step = y[step]

            loss, acc, global_step = swwae.train(X_step,y_step)

            total_loss += loss
            total_accuracy += acc
            epoch_loss += loss
            epoch_acc += acc

            if (step + 1) % parsed.info_step == 0:
                avg_loss = total_loss / parsed.info_step
                avg_acc = total_accuracy / parsed.info_step
                save_loss(avg_loss)

                total_loss_test = 0.0
                total_acc_test = 0.0
                test_steps = len(X_test)

                for test_step in range(test_steps):
                    X_test_step = X_test[test_step]
                    y_test_step = y_test[test_step]

                    loss, acc = swwae.eval(input=X_test_step, labels=y_test_step)

                    total_loss_test += loss
                    total_acc_test += acc

                print("Test average loss: {}, average acc: {}".format(total_loss_test / test_steps, total_acc_test / test_steps))

                print("Train epoch {}:\n\tstep {}\n\tAvg Loss: {} Avg accuracy {}".format(e + 1, step + 1, avg_loss, avg_acc),
                      flush=True)

                total_loss = 0.0
                total_accuracy = 0.0
                end = time.time()
                hours, rem = divmod(end - start, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Elapsed: {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)), flush=True)
                start = time.time()

            if parsed.save_step is not None:
                if (global_step + 1) % parsed.save_step == 0:
                    swwae.save(path=parsed.output_dir)

        print("Train epoch {}: avg. loss: {}, avg. acc: {}".format(e + 1, epoch_loss / train_steps, epoch_acc / train_steps), flush=True)
        X, y = dataset.get_batches(parsed.batch_size)

    if parsed.output_dir is not None:
        swwae.save(path=parsed.output_dir)

    print("Starting test..")

    total_loss = 0.0
    total_acc = 0.0
    test_steps = len(X_test)

    for test_step in range(test_steps):
        X_test_step = X_test[test_step]
        y_test_step = y_test[test_step]

        loss, acc = swwae.eval(input=X_test_step,labels=y_test_step)

        total_loss += loss
        total_acc += acc

    print("Final Test average loss: {}, average acc: {}".format(total_loss/test_steps, total_acc/test_steps))


if __name__ == "__main__":
   main()
