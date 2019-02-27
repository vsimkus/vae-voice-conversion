import argparse
import json


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Voice conversion parameter settings.')

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--gpu_id', type=str, default="None", help="A comma-separated string indicating the gpus to use")
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--learning_rate', nargs="?", type=float, default=1e-03,
                        help='Learning rate to use for Adam')
    parser.add_argument('--filepath_to_arguments_json_file', nargs="?", type=str, default=None,
                        help='All configs and model-specific parameters can be set in a json file. If a setting is provided in the json file it will be prefered.')
    parser.add_argument('--dataset_root_path', type=str, default="data")    

    args = parser.parse_args()
    gpu_id = str(args.gpu_id)
    if args.filepath_to_arguments_json_file is not None:
        args = extract_args_from_json(json_file_path=args.filepath_to_arguments_json_file, existing_args_dict=args)

    if gpu_id != "None":
        args.gpu_id = gpu_id

    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)
    return args


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

    @staticmethod
    def from_nested_dict(data):
        """ Construct nested AttributeAccessibleDict from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttributeAccessibleDict(
                {key: AttributeAccessibleDict.from_nested_dict(data[key])
                    for key in data})


def extract_args_from_json(json_file_path, existing_args_dict=None):

    summary_filename = json_file_path
    with open(summary_filename) as f:
        arguments_dict = json.load(fp=f)

    if existing_args_dict is not None:
        for key, value in vars(existing_args_dict).items():
            if key not in arguments_dict:
                arguments_dict[key] = value

    arguments_dict = AttributeAccessibleDict.from_nested_dict(arguments_dict)

    return arguments_dict
