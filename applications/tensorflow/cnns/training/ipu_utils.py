# Copyright 2019 Graphcore Ltd.
from tensorflow.python.ipu import utils


def get_config(prng=False,
               ipu_id=-1,
               shards=1,
               number_of_replicas=1,
               max_cross_replica_buffer_size=10*1024*1024,
               merge_infeed_io_copies=True,
               fp_exceptions=True,
               xla_recompute=False,
               seed=None,
               availableMemoryProportion=None,
               selection_order=None,
               max_report_size=268435456,
               use_poplar_text_report=False,
               report_directory='',
               profiling=False,
               profile_execution=False):
    """Builds ipu_options"""
    config = utils.create_ipu_config(max_cross_replica_sum_buffer_size=max_cross_replica_buffer_size,
                                     merge_infeed_io_copies=merge_infeed_io_copies,
                                     always_rearrange_copies_on_the_host=True,
                                     selection_order=selection_order,
                                     max_report_size=max_report_size,
                                     use_poplar_text_report=use_poplar_text_report,
                                     report_directory=report_directory,
                                     profiling=profiling,
                                     profile_execution=profile_execution)
    if ipu_id == -1:
        config = utils.auto_select_ipus(config, number_of_replicas*shards)
    else:
        config = utils.select_ipus(config, [ipu_id])
    config = utils.set_compilation_options(config, {
        "device.clearAtomicFlagAfterExchange": "false",
        "prng.enable": "true" if prng else "false",
        "target.deterministicWorkers": "false" if seed is None else "true",
    })

    if availableMemoryProportion is not None:
        config = utils.set_convolution_options(config, {
            "availableMemoryProportion": str(availableMemoryProportion)
        })

    if xla_recompute:
        utils.set_recomputation_options(config, allow_recompute=True)

    config = utils.set_floating_point_behaviour_options(config, inv=fp_exceptions, div0=fp_exceptions,
                                                        oflo=fp_exceptions, esr=prng, nanoo=True)
    config = utils.set_ipu_model_options(config, compile_ipu_code=False)
    return config
