from datapy.data.datasets import CIFAR10Dataset, MNISTDataset, FashionDataset, STLDataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_parser
from utils import save_loss, clear_loss

def main():
    clear_loss()
    parser = get_parser()
    parsed = parser.parse_args()

    assert ((parsed.output_dir is None and parsed.save_step is None) or
            (parsed.output_dir is not None and parsed.save_step is not None)), "Save step and output directory must be " \
                                                                               "null at the same time or not null at the same time"

    ds_type = parsed.dataset

    if ds_type == 'cifar10':
        dataset = CIFAR10Dataset()
        dataset.process()
        img_shape = [32,32,3]
    elif ds_type == 'mnist':
        dataset = MNISTDataset()
        dataset.process()
        img_shape = [28,28,1]
    elif parsed.dataset == 'fashion':
        dataset = FashionDataset()
        dataset.process()
        img_shape = [28, 28, 1]
    elif parsed.dataset == 'stl10':
        dataset = STLDataset(is_ae=True)
        dataset.process()
        img_shape = [96, 96, 3]
    else:
        print("Unknown dataset")
        exit()

    layers = parse_layers(parsed.layer_str)
    fc_size = parsed.fc_layers

    sess = tf.Session()
    swwae = SWWAE(sess,img_shape,'autoencode',layers,learning_rate=parsed.learning_rate,lambda_rec=parsed.lambda_rec,
                  lambda_M=parsed.lambda_M,dtype=tf.float32, tensorboard_id=parsed.tensorboard_id, encoder_train=True,
                  rep_size=fc_size, batch_size=parsed.batch_size, sparsity=parsed.sparsity, beta=parsed.beta)

    if parsed.rest_dir is not None:
        swwae.restore(parsed.rest_dir)
    X_train, X_actual = dataset.get_batches_actual(parsed.batch_size)
    X_test, _ = dataset.get_batches(parsed.batch_size, train=False)
    test_steps = len(X_test)

    train_steps = int(len(dataset.training_data) / parsed.batch_size)

    print("Started training.\nTrain steps: {}".format(train_steps))

    for e in range(parsed.num_epochs):
        total_loss = 0.0
        epoch_loss = 0.0
        batches = 0
        for i, x_batch in enumerate(X_train):
            loss, global_step = swwae.train(x_batch, X_actual[i])
            batches += 1

            total_loss += loss
            epoch_loss += loss

            if (batches + 1) % parsed.info_step == 0:
                avg_loss = total_loss / parsed.info_step
                save_loss(avg_loss)

                for j, x_batch_test in enumerate(X_test):
                    swwae.eval(x_batch_test, x_batch_test)

                #print("Train epoch {}:\n\tstep {}\n\tavg. L2 Loss: {}".format(e + 1, step + 1, avg_loss),
                 #     flush=True)

                total_loss = 0.0

            if parsed.save_step is not None:
                if (global_step + 1) % parsed.save_step == 0:
                    swwae.save(path=parsed.output_dir)

            if batches >= train_steps:
                break

        print("Train epoch {}: avg. loss: {}".format(e + 1, epoch_loss / train_steps), flush=True)


    if parsed.output_dir is not None:
        swwae.save(path=parsed.output_dir)

    print("Starting test..")

    total_loss = 0.0

    for test_step in range(test_steps):
        X_test_step = X_test[test_step]

        loss = swwae.eval(input=X_test_step)

        total_loss += loss

    print("Test average loss: {}".format(total_loss/test_steps))


if __name__ == "__main__":
   main()