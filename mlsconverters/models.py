# -*- coding: utf-8 -*-
#
# Copyright 2020 - Viktor Gal
# A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and
# Eidgenössische Technische Hochschule Zürich (ETHZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import calamus.fields as fields
import marshmallow.fields as msmlfields
from calamus.fields import _JsonLDField
from calamus.schema import JsonLDSchema, blank_node_id_strategy

ML_SCHEMA = fields.Namespace("http://www.w3.org/ns/mls#")
XML_SCHEMA = fields.Namespace("http://www.w3.org/2001/XMLSchema#")
DC_TERMS = fields.Namespace("http://purl.org/dc/terms/")
RDFS = fields.Namespace("http://www.w3.org/2000/01/rdf-schema#")


class ParameterValue(_JsonLDField, msmlfields.Field):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        value = super()._serialize(value, attr, obj, **kwargs)
        if self.parent.opts.add_value_types:
            xsd_type = "xsd:anyURI"
            if type(value) == bool:
                xsd_type = "xsd:boolean"
            elif type(value) == int:
                xsd_type = "xsd:int"
            elif type(value) == float:
                xsd_type = "xsd:float"
            elif type(value) == str:
                xsd_type = "xsd:string"
            value = {"@type": xsd_type, "@value": value}
        return value

    def _deserialize(self, value, attr, data, **kwargs):
        v = json.loads(value)
        return v["@value"]


class EvaluationMeasure:
    def __init__(self, _id):
        self._id = _id


class EvaluationMeasureSchema(JsonLDSchema):
    _id = fields.Id()

    class Meta:
        rdf_type = ML_SCHEMA.EvaluationMeasure
        model = EvaluationMeasure


class ModelEvaluation:
    def __init__(self, _id, value, specified_by):
        self._id = _id
        self.value = value
        self.specified_by = specified_by


class ModelEvaluationSchema(JsonLDSchema):
    _id = fields.Id()
    value = ParameterValue(ML_SCHEMA.hasValue)
    specified_by = fields.Nested(ML_SCHEMA.specifiedBy, EvaluationMeasureSchema)

    class Meta:
        rdf_type = ML_SCHEMA.ModelEvaluation
        model = ModelEvaluation


class HyperParameter:
    def __init__(self, _id, model_hash):
        self._id = "http://www.w3.org/ns/mls#HyperParameter.{}.{}".format(
            _id, model_hash
        )
        self.label = _id


class HyperParameterSchema(JsonLDSchema):
    _id = fields.Id()
    label = fields.String(RDFS.label)

    class Meta:
        rdf_type = ML_SCHEMA.HyperParameter
        model = HyperParameter


class Algorithm:
    def __init__(self, _id):
        self._id = _id
        self.label = _id


class AlgorithmSchema(JsonLDSchema):
    _id = fields.Id()
    label = fields.String(RDFS.label)

    class Meta:
        rdf_type = ML_SCHEMA.Algorithm
        model = Algorithm


class HyperParameterSetting:
    def __init__(self, value, specified_by, model_hash):
        self._id = f"http://www.w3.org/ns/mls#HyperParameterSetting.{specified_by.label}.{model_hash}"
        self.value = value
        self.specified_by = specified_by


class HyperParameterSettingSchema(JsonLDSchema):
    _id = fields.Id()
    value = ParameterValue(ML_SCHEMA.hasValue)
    specified_by = fields.Nested(
        ML_SCHEMA.specifiedBy, HyperParameterSchema, only=("_id",)
    )

    class Meta:
        rdf_type = ML_SCHEMA.HyperParameterSetting
        model = HyperParameterSetting
        add_value_types = True


class Implementation:
    """Repesent an ML Schema defined Model."""

    def __init__(self, _id, parameters, implements=None, version=None, name=None):
        self._id = _id
        self.name = name
        self.parameters = parameters
        self.implements = implements
        self.version = version


class ImplementationSchema(JsonLDSchema):
    _id = fields.Id()
    name = fields.String(DC_TERMS.title)
    parameters = fields.Nested(
        ML_SCHEMA.hasHyperParameter, HyperParameterSchema, many=True
    )
    implements = fields.Nested(ML_SCHEMA.implements, AlgorithmSchema)
    version = fields.String(DC_TERMS.hasVersion)

    class Meta:
        rdf_type = ML_SCHEMA.Implementation
        model = Implementation


class Run:
    def __init__(
        self,
        _id,
        executes=None,
        input_values=[],
        output_values=[],
        realizes=None,
        version=None,
        name=None,
    ):
        self._id = _id
        self.executes = executes
        self.input_values = input_values
        self.output_values = output_values
        self.realizes = realizes
        self.version = version
        self.name = name


class RunSchema(JsonLDSchema):
    _id = fields.Id()
    executes = fields.Nested(ML_SCHEMA.executes, ImplementationSchema)
    input_values = fields.Nested(
        ML_SCHEMA.hasInput, HyperParameterSettingSchema, many=True, flattened=True
    )
    output_values = fields.Nested(ML_SCHEMA.hasOutput, ModelEvaluationSchema, many=True)
    realizes = fields.Nested(ML_SCHEMA.implements, AlgorithmSchema)
    version = fields.String(DC_TERMS.hasVersion)
    name = fields.String(DC_TERMS.title)

    class Meta:
        rdf_type = ML_SCHEMA.Run
        model = Run
