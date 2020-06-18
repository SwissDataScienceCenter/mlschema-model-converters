from mls.models import Algorithm, HyperParameter, HyperParameterSetting, Implementation, Run, ModelEvaluation
import autosklearn.automl
import autosklearn.pipeline
import json
import numpy as np
from scipy.stats._distn_infrastructure import rv_frozen
from functools import partial

from .sklearn import deep_get_params
from ConfigSpace.configuration_space import Configuration

def to_mls(autosklearn_model: autosklearn.automl.AutoML):

    def blank_node(id):
        return "_:{}".format(id)

    def custom_types(v):
        if isinstance(v, np.random.mtrand.RandomState):
            return v.get_state()
        elif isinstance(v, Configuration):
            return v.get_dictionary()
        elif isinstance(v, autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice):
            v = v.get_components()
            from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
            return {k: x.__name__ if issubclass(x, AutoSklearnPreprocessingAlgorithm) else x for k, x in v.items()}
        elif isinstance(v, autosklearn.pipeline.components.classification.ClassifierChoice):
            v = v.get_components()
            from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
            return {k: x.__name__ if issubclass(x, AutoSklearnClassificationAlgorithm) else x for k, x in v.items()}
        else:
            return v

    deep_get_params_ = partial(deep_get_params, custom_type_handler=custom_types)

    params = deep_get_params_(autosklearn_model)

    model_class = "_:{}.{}".format(type(autosklearn_model).__module__, type(autosklearn_model).__name__)

    model = Run()

    implementation = Implementation(id=model_class)
    implementation.parameters = [HyperParameter(id=blank_node(key)) for key in params.keys()]
    implementation.implements = Algorithm(id=model_class)
    model.executes = implementation

    model.input_values = [
        HyperParameterSetting(value=val, specified_by={"@id": blank_node(key)})
        for key, val in params.items() if val is not None
    ]

    model.output_values = [
        ModelEvaluation(value=deep_get_params_(automl_model),
                        specified_by={"@id": blank_node('automl')})
        for automl_model in autosklearn_model.get_models_with_weights()
    ]

    return model.asjsonld()
