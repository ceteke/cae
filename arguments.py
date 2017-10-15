import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='SWWAE Implementation')

    parser.add_argument("-e", "--epochs", help="Number of epochs", required=True, dest='num_epochs', type=int)
    parser.add_argument("-ng", "--num_gpu", help="Number of GPUs", required=True, dest='num_gpu', type=int)
    parser.add_argument("-lh", "--log_h", help="Log histograms", required=True, dest='log_h', type=int)
    parser.add_argument("-i", "--info", help="Info steps", required=True, dest='info_step', type=int)
    parser.add_argument("-b", "--batch", help="Batch size", required=True, dest='batch_size', type=int)
    parser.add_argument("-l", "--layers", help="Layer string. See README.md", required=False, dest='layer_str', type=str,
                        default='(64)3c-4p-(64)3c-3p-(128)3c-(128)3c-2p-(256)3c-(256)3c-(256)3c-(512)3c-(512)3c-(512)3c-2p')
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", required=True, dest='learning_rate', type=float)


    parser.add_argument("-lm", "--lambda_M", help="Middle loss coefficient", required=False, dest='lambda_M',
                        type=float, default=0.2)
    parser.add_argument("-lrec", "--lambda_rec", help="Reconstruction loss coefficient", required=False, dest='lambda_rec',
                        type=float, default=1.0)
    parser.add_argument("-tb", "--tensorboard", help="ID of tensorboard", required=False, dest='tensorboard_id',
                        type=int)
    parser.add_argument("-o", "--output", help="Output directory [optional]", required=False, dest='output_dir', type=str)
    parser.add_argument("-fc", "--full_connected", help="Fully connected layers", required=False, dest='fc_layers',
                        type=str)
    parser.add_argument("-s", "--save", help="Save steps [optional]", required=False, dest='save_step', type=int)
    parser.add_argument("-ds", "--dataset", help="Which dataset", required=True, dest='dataset',
                        type=str)
    parser.add_argument("-r", "--restor_dir", help="Restore directory", required=False, dest='rest_dir',
                        type=str)

    return parser

def get_emb_parser():
    parser = argparse.ArgumentParser(description='SWWAE Embedding')
    parser.add_argument("-b", "--batch", help="Batch size", required=True, dest='batch_size', type=int)
    parser.add_argument("-o", "--output", help="Output directory", required=True, dest='out_dir', type=str)
    parser.add_argument("-l", "--layers", help="Layer string. See README.md", required=False, dest='layer_str',
                        type=str,
                        default='(64)3c-4p-(64)3c-3p-(128)3c-(128)3c-2p-(256)3c-(256)3c-(256)3c-(512)3c-(512)3c-(512)3c-2p')

    parser.add_argument("-s", "--save", help="Save embedding path", required=False, dest='save_path',
                        type=str, default='embedding')
    parser.add_argument("-fc", "--full_connected", help="Fully connected layers", required=False, dest='fc_layers',
                        type=str)
    parser.add_argument("-ds", "--dataset", help="Which dataset", required=True, dest='dataset',
                        type=str)

    return parser

def get_class_parser():
    parser = argparse.ArgumentParser(description='SWWAE Classification Parser')
    parser.add_argument("-e", "--epochs", help="Number of epochs", required=True, dest='num_epochs', type=int)
    parser.add_argument("-i", "--info", help="Info steps", required=True, dest='info_step', type=int)
    parser.add_argument("-b", "--batch", help="Batch size", required=True, dest='batch_size', type=int)
    parser.add_argument("-o", "--output", help="Output directory", required=True, dest='output_dir', type=str)
    parser.add_argument("-s", "--save", help="Save steps [optional]", required=False, dest='save_step', type=int)
    parser.add_argument("-m", "--model", help="Load directory", required=True, dest='load_dir', type=str)
    parser.add_argument("-et", "--encoder_train", help="Train encoder", required=True, dest='encoder_train', type=int)
    parser.add_argument("-l", "--layers", help="Layer string. See README.md", required=False, dest='layer_str',
                        type=str,
                        default='(64)3c-4p-(64)3c-3p-(128)3c-(128)3c-2p-(256)3c-(256)3c-(256)3c-(512)3c-(512)3c-(512)3c-2p')
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", required=True, dest='learning_rate', type=float)
    parser.add_argument("-tb", "--tensorboard", help="ID of tensorboard", required=False, dest='tensorboard_id',
                        type=int)
    parser.add_argument("-fc", "--fully_connected", help="Fully connected layers", required=True, dest='fc_layers', type=str)
    return parser