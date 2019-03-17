import argparse
import json
import os
import sys
import GPUtil

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
    parser.add_argument('--tuning_mode', type=str2bool, default=False, help='Use a smaller dataset split, and discard most of data. Use this for faster feedback and initial tuning of the algorithms')
    parser.add_argument('--print_timings', type=str2bool, default=False, help='Print timings during different stages of training.')

    parser.add_argument('--eval_audio_path', nargs='?', type=str, help='[Evaluation] Path to audio file for producing reconstruction.')
    parser.add_argument('--eval_out_path', nargs='?', type=str, help='[Evaluation] Path to output file for producing reconstruction.')
    parser.add_argument('--eval_speaker_id', nargs='?', type=int, help='[Evaluation] Speaker id for speaker conversion.')
    parser.add_argument('--dataset', type=str, default='VCCWORLD2016', help='VCCWORLD2016, VCCRaw2016 or VCTK')

    args = parser.parse_args()
    gpu_id = str(args.gpu_id)
    if args.filepath_to_arguments_json_file is not None:
        args = extract_args_from_json(json_file_path=args.filepath_to_arguments_json_file, existing_args_dict=args)

    if gpu_id != "None":
        args.gpu_id = gpu_id

    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)

    # Temporary solution for incorrect SLURM config
    if args.use_gpu == True:
        num_requested_gpus = len(args.gpu_id.split(","))
        num_received_gpus = len(GPUtil.getAvailable(order='first', limit=8, maxLoad=0.1,
                                             maxMemory=0.1, includeNan=False,
                                             excludeID=[], excludeUUID=[]))

        if num_requested_gpus == 1 and num_received_gpus > 1:
            print("Detected Slurm problem with GPUs, attempting automated fix")
            gpu_to_use = GPUtil.getAvailable(order='first', limit=num_received_gpus, maxLoad=0.1,
                                             maxMemory=0.1, includeNan=False,
                                             excludeID=[], excludeUUID=[])
            if len(gpu_to_use) > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use[0])
                print("Using GPU with ID", gpu_to_use[0])
            else:
                print("Not enough GPUs available, please try on another node now, or retry on this node later")
                sys.exit()

        elif num_requested_gpus > 1 and num_received_gpus > num_requested_gpus:
            print("Detected Slurm problem with GPUs, attempting automated fix")
            gpu_to_use = GPUtil.getAvailable(order='first', limit=num_received_gpus,
                                             maxLoad=0.1,
                                             maxMemory=0.1, includeNan=False,
                                             excludeID=[], excludeUUID=[])

            if len(gpu_to_use) >= num_requested_gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_idx) for gpu_idx in gpu_to_use[:num_requested_gpus])
                print("Using GPU with ID", gpu_to_use[:num_requested_gpus])
            else:
                print("Not enough GPUs available, please try on another node now, or retry on this node later")
                sys.exit()


    import torch
    args.use_cuda = torch.cuda.is_available()

    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device = torch.cuda.current_device()
        print("use {} GPU(s)".format(torch.cuda.device_count()), file=sys.stderr)
    else:
        print("use CPU", file=sys.stderr)
        device = torch.device('cpu')  # sets the device to be CPU

    return args, device


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
