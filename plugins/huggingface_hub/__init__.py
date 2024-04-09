"""Hugging Face Hub plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types


class PushToHuggingFaceHub(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="push_to_huggingface_hub",
            label="Push dataset to Hugging Face Hub",
            dynamic=True,
        )
        # _config.icon = "./assets/icon.svg" # TODO: add icon that WORKS
        return _config

    def resolve_input(self, ctx):
        pass

    def resolve_output(self, ctx):
        pass

    def execute(self, ctx):
        pass


class LoadFromHuggingFaceHub(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="load_from_huggingface_hub",
            label="Load dataset from Hugging Face Hub",
            dynamic=True,
        )
        # _config.icon = "./assets/icon.svg" # TODO: add icon that WORKS
        return _config

    def resolve_input(self, ctx):
        pass

    def resolve_output(self, ctx):
        pass

    def execute(self, ctx):
        pass


def register(plugin):
    plugin.register(PushToHuggingFaceHub)
    plugin.register(LoadFromHuggingFaceHub)

