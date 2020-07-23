import inspect
import numpy as np
from .models import HyperParameter, HyperParameterSetting
from uuid import uuid1


def _jsonize_value(value):
    """
    JSON dump requires primitive types
    """
    value_type = type(value)
    if value_type == np.float32 or value_type == np.float64:
        return float(value)
    elif value_type == np.int32 or value_type == np.int64:
        return int(value)
    return value


def mls_params(params):
    mls_parameters = []
    mls_input_values = []
    for key, value in params.items():
        hp = HyperParameter(_id=key)
        mls_parameters.append(hp)
        if value is not None:
            mls_input_values.append(
                HyperParameterSetting(value=_jsonize_value(val), specified_by=hp)
            )
    return parameters, mls_input_values


def mls_param(key, value):
    hp = HyperParameter(_id=key)
    return hp, HyperParameterSetting(
        value=xsd_type(_jsonize_value(value)), specified_by=hp)


# TODO: once PR #1 merged this should be dropped
def mls_add_param(mls, key, value):
    p, iv = mls_param(key, value)
    mls.executes.parameters.append(p)
    mls.input_values.append(iv)


def mls_add_params(mls, key, value):
    for key, value in params.items():
        hp = HyperParameter(_id=key)
        mls.executes.parameters.append(hp)
        if value is not None:
            mls.input_values.append(
                HyperParameterSetting(
                    value=xsd_type(_jsonize_value(val)), specified_by=hp)
            )


def xsd_type(v):
    xsd_type = "xsd:anyURI"
    if type(v) == bool:
        xsd_type = "xsd:boolean"
    elif type(v) == int:
        xsd_type = "xsd:int"
    elif type(v) == float:
        xsd_type = "xsd:float"
    elif type(v) == str:
        xsd_type = "xsd:string"
    return {'@type': xsd_type, '@value': v}


def get_unspecified_default_args(user_args, user_kwargs, all_param_names, all_default_values):
    num_args_without_default_value = len(all_param_names) - len(all_default_values)

    # all_default_values correspond to the last len(all_default_values) elements of the arguments
    default_param_names = all_param_names[num_args_without_default_value:]

    default_args = dict(zip(default_param_names, all_default_values))

    # The set of keyword arguments that should not be logged with default values
    user_specified_arg_names = set(user_kwargs.keys())

    num_user_args = len(user_args)

    # This checks if the user passed values for arguments with default values
    if num_user_args > num_args_without_default_value:
        num_default_args_passed_as_positional = num_user_args - num_args_without_default_value
        # Adding the set of positional arguments that should not be logged with default values
        names_to_exclude = default_param_names[:num_default_args_passed_as_positional]
        user_specified_arg_names.update(names_to_exclude)

    return {name: value for name, value in default_args.items()
            if name not in user_specified_arg_names}


def fn_args_as_params(fn, args, kwargs, unlogged=[]):  # pylint: disable=W0102
    # all_default_values has length n, corresponding to values of the
    # last n elements in all_param_names
    pos_params, _, _, pos_defaults, kw_params, kw_defaults, _ = inspect.getfullargspec(fn)

    kw_params = list(kw_params) if kw_params else []
    pos_defaults = list(pos_defaults) if pos_defaults else []
    all_param_names = pos_params + kw_params
    all_default_values = pos_defaults + [kw_defaults[param] for param in kw_params]

    params = []
    input_values = []
    # Checking if default values are present for logging. Known bug that getargspec will return an
    # empty argspec for certain functions, despite the functions having an argspec.
    if all_default_values is not None and len(all_default_values) > 0:
        # Logging the default arguments not passed by the user
        defaults = get_unspecified_default_args(args, kwargs, all_param_names, all_default_values)

        for name in [name for name in defaults.keys() if name in unlogged]:
            del defaults[name]
        p, iv = mls_params(args_dict)
        params.append(p)
        input_values.append(iv)

    # Logging the arguments passed by the user
    args_dict = dict((param_name, param_val) for param_name, param_val
                     in zip(all_param_names, args)
                     if param_name not in unlogged)


    if args_dict:
        p, iv = mls_params(args_dict)
        params.append(p)
        input_values.append(iv)

    # Logging the kwargs passed by the user
    for param_name in kwargs:
        if param_name not in unlogged:
            p, iv = mls_param(param_name, kwargs[param_name])
            params.append(p)
            input_values.append(iv)

    return params, input_values
