_params: ParameterRecord | None = None
_experiment_id = trial_env_vars.NNI_EXP_ID or 'STANDALONE'
_trial_id = trial_env_vars.NNI_TRIAL_JOB_ID or 'STANDALONE'
_sequence_id = int(trial_env_vars.NNI_TRIAL_SEQ_ID) if trial_env_vars.NNI_TRIAL_SEQ_ID is not None else 0


def report_intermediate_result(metric: TrialMetric | dict[str, Any]) -> None:
    """
    Reports intermediate result to NNI.

    ``metric`` should either be a float, or a dict that ``metric['default']`` is a float.

    If ``metric`` is a dict, ``metric['default']`` will be used by tuner,
    and other items can be visualized with web portal.

    Typically ``metric`` is per-epoch accuracy or loss.

    Parameters
    ----------
    metric : :class:`~nni.typehint.TrialMetric`
        The intermeidate result.
    """
    global _intermediate_seq
    assert _params or trial_env_vars.NNI_PLATFORM is None, \
        'nni.get_next_parameter() needs to be called before report_intermediate_result'
    get_default_trial_command_channel().send_metric(
        parameter_id=_params['parameter_id'] if _params else None,
        trial_job_id=trial_env_vars.NNI_TRIAL_JOB_ID,
        type='PERIODICAL',
        sequence=_intermediate_seq,
        value=cast(TrialMetric, metric)
    )
    _intermediate_seq += 1