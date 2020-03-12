# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
from .resnet_base import *



class ResNet(ResNetBase):
    def __init__(self, opts, is_training=True):
        if opts['dataset'] == 'imagenet':
            definitions = {**RESNETS_Imagenet, **RESNETS_Bottleneck_Imagenet}
        else:
            if opts['widenet']:
                definitions = {
                    **RESNETS_Cifar,
                    **RESNETS_Bottleneck_Cifar_wide
                }
            else:
                definitions = {**RESNETS_Cifar, **RESNETS_Bottleneck_Cifar}
        definition = definitions[opts["model_size"]]
        super().__init__(opts, definition, conv, is_training)
        self.block_fn = partial(definition.block_fn,
                                shortcut_type=opts["shortcut_type"],
                                conv=self.conv,
                                norm=self.norm)


def Model(opts, training, image):
    return ResNet(opts, training)(image)


def staged_model(opts):
    splits = opts['pipeline_splits']
    x = ResNet(opts, True)
    possible_splits = [
        s.keywords['name'] for s in x._build_function_list()
        if 'relu' in s.keywords['name']
    ]
    print('possible_splits={}'.format(possible_splits))
    if splits is None:
        possible_splits = [
            s.keywords['name'] for s in x._build_function_list()
            if 'relu' in s.keywords['name']
        ]
        raise ValueError(
            "--pipeline-splits not specified. Need {} of {}".format(
                opts['shards'] - 1, possible_splits))
    splits.append(None)
    stages = [partial(x.first_stage, first_split_name=splits[0])]
    for i in range(len(splits) - 1):
        stages.append(
            partial(x.later_stage,
                    prev_split_name=splits[i],
                    end_split_name=splits[i + 1]))
    return stages


def add_arguments(parser):
    group = parser.add_argument_group('ResNet')
    add_resnet_arguments(group)
    return parser


def set_defaults(opts):
    opts['summary_str'] += "ResNet-{model_size}\n"
    set_resnet_defaults(opts, "RN")
