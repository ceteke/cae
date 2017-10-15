from datapy.data.datasets import CIFAR10Dataset, MNISTDataset, FashionDataset, STLDataset
from model import SWWAE
import tensorflow as tf
from utils import parse_layers
from arguments import get_parser
from utils import save_loss, clear_loss
import time
from keras.preprocessing.image import ImageDataGenerator

def main():
    clear_loss()
    parser = get_parser()
    parsed = parser.parse_args()

    assert ((parsed.output_dir is None and parsed.save_step is None) or
            (parsed.output_dir is not None and parsed.save_step is not None)), "Save step and output directory must be " \
                                                                               "null at the same time or not null at the same time"

    ds_type = parsed.dataset

    log_h = parsed.loh_h == 1

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
    if parsed.fc_layers is not None:
        fc_layers = [int(x) for x in parsed.fc_layers.split('-')]
    else:
        fc_layers = []

    sess = tf.Session()
    swwae = SWWAE(sess,img_shape,'autoencode',layers,learning_rate=parsed.learning_rate,lambda_rec=parsed.lambda_rec,
                  lambda_M=parsed.lambda_M,dtype=tf.float32, tensorboard_id=parsed.tensorboard_id, encoder_train=True,
                  fc_ae_layers=fc_layers, batch_size=parsed.batch_size, log_h=log_h)

    if parsed.rest_dir is not None:
        swwae.restore(parsed.rest_dir)

    X_test, _ = dataset.get_batches(parsed.batch_size, train=False)
    test_steps = len(X_test)

    print("Preprocessing")
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0.0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(dataset.training_data)

    train_steps = int(len(dataset.training_data) / parsed.batch_size)

    print("Started training.\nTrain steps: {}".format(train_steps))

    for e in range(parsed.num_epochs):
        total_loss = 0.0
        epoch_loss = 0.0
        batches = 0
        start_time = time.time()
        for x_batch in datagen.flow(dataset.training_data, batch_size=parsed.batch_size):
            loss, global_step = swwae.train(x_batch)
            batches += 1

            total_loss += loss
            epoch_loss += loss

            if (batches + 1) % parsed.info_step == 0:
                avg_loss = total_loss / parsed.info_step
                save_loss(avg_loss)

                for test_step in range(test_steps):
                    X_test_step = X_test[test_step]
                    swwae.eval(input=X_test_step)

                total_loss = 0.0

            if parsed.save_step is not None:
                if (global_step + 1) % parsed.save_step == 0:
                    swwae.save(path=parsed.output_dir)

            if batches >= train_steps:
                break
        elapsed_time = time.time() - start_time
        print("Train epoch {}: avg. loss: {}, elapsed {}".format(e + 1, epoch_loss / train_steps, elapsed_time), flush=True)

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