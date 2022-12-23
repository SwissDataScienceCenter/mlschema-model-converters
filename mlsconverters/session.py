from uuid import uuid1

from . import io
from .common import generate_unique_id
from .models import (Algorithm, EvaluationMeasure, HyperParameter,
                     HyperParameterSetting, Implementation, ModelEvaluation,
                     Run, RunSchema)


class Session:
    def __init__(self, name, run_id=None):
        self._name = name
        self._run_id = uuid1().fields[0] if run_id is None else run_id
        self._run = Run(self._run_id)
        self._run.realizes = Algorithm(self._name)
        self._hp = dict()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        params = []
        for k, v in self._hp.items():
            hp = HyperParameter(k, model_hash=self._run._id)
            params.append(hp)
            self._run.input_values.append(
                HyperParameterSetting(v, hp, model_hash=self._run._id)
            )
        self._run.executes = Implementation(
            generate_unique_id("http://www.w3.org/ns/mls#Implementation"),
            params,
            implements=self._run.realizes,
        )
        io.log_renku_mls(RunSchema().dumps(self._run), str(self._run_id), force=True)

    def param(self, param_name, value):
        self._hp.update({param_name: value})

    def params(self, params):
        self._hp.update(params)

    def metric(self, metric_name, value):
        _id = generate_unique_id("http://www.w3.org/ns/mls#ModelEvaluation")
        self._run.output_values.append(
            ModelEvaluation(
                _id=_id,
                value=value,
                specified_by=EvaluationMeasure(
                    _id="http://www.w3.org/ns/mls#{}".format(metric_name)
                ),
            )
        )
