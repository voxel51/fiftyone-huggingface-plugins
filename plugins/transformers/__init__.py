"""Hugging Face Transformers plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import importlib

import fiftyone as fo
import fiftyone.core.utils as fou
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.zoo as foz

hfh = fou.lazy_import("huggingface_hub")
transformers = fou.lazy_import("transformers")


FIFTYONE_HUB_URL_TEMPLATE = (
    "https://huggingface.co/datasets?other=fiftyone&sort=trending&p={i}"
)

LICENSES = (
    "apache-2.0",
    "cc",
    "cc0-1.0",
    "cc-by-2.0",
    "cc-by-2.5",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "cc-by-nc-2.0",
    "cc-by-nc-3.0",
    "cc-by-nc-4.0",
    "c-uda",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "gpl",
    "gpl-2.0",
    "gpl-3.0",
    "mit",
    "odc-by",
    "unlicense",
    "other",
)

PREVIEW_EXTS = (".png", ".jpg", ".jpeg", ".gif")


task_map = {
    "Image Classification": ["ForImageClassification"],
    "Object Detection": ["ForObjectDetection"],
    "Depth Estimation": ["ForDepthEstimation"],
    "Image Segmentation": [
        "ForSemanticSegmentation",
        "ForInstanceSegmentation",
        "ForSegmentation",
        "ForUniversalSegmentation",
    ],
}

task_to_zoo_map = {
    "Image Classification": "classification-transformer-torch",
    "Object Detection": "detection-transformer-torch",
    "Depth Estimation": "depth-estimation-transformer-torch",
    "Image Segmentation": "segmentation-transformer-torch",
}


def _get_model_arch_for_pattern(task_pattern):
    all_models = dir(transformers.models)

    architecture_strings = []
    for model_name in all_models:
        try:
            # Dynamically import the module
            module = importlib.import_module(
                f"transformers.models.{model_name}"
            )
            attrs = dir(module)
            for attr in attrs:
                if attr.endswith(task_pattern) and not attr.startswith(
                    "AutoModel"
                ):
                    architecture_strings.append(attr)

        except:
            continue

    return architecture_strings


def get_model_architectures(task):
    task_patterns = task_map[task]
    all_architectures = []
    for task_pattern in task_patterns:
        architectures = _get_model_arch_for_pattern(task_pattern)
        all_architectures.extend(architectures)

    return sorted(list(set(all_architectures)))


def get_model_names(architecture_string):
    model_class = getattr(transformers, architecture_string)
    model_name = model_class.base_model_prefix.replace("_", "-")
    api = hfh.HfApi()
    models = api.list_models(model_name=model_name, limit=25)
    model_names = [model.modelId for model in models]
    return model_names


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


def _get_fields_with_type(view, type):
    if issubclass(type, fo.Field):
        return list(view.get_field_schema(ftype=type).keys())

    return list(view.get_field_schema(embedded_doc_type=type).keys())


def _apply_transformer_model_inputs(ctx, inputs):
    task_group = types.RadioGroup()

    for task in task_map.keys():
        task_group.add_choice(task, label=task)

    inputs.enum(
        "task",
        task_group.values(),
        label="Task",
        description="Select the task for which you want to apply the model",
        view=types.DropdownView(),
        required=False,
    )

    task = ctx.params.get("task", None)
    if task is None:
        inputs.str("no_task", required=True, view=types.HiddenView())
        return inputs

    architectures = get_model_architectures(task)
    arch_group = types.RadioGroup()

    for arch in architectures:
        arch_group.add_choice(arch, label=arch)

    inputs.enum(
        "architecture",
        arch_group.values(),
        label="Architecture",
        description="Select the architecture of the model",
        view=types.DropdownView(),
        required=False,
    )

    arch = ctx.params.get("architecture", None)
    if arch is None:
        inputs.str(
            "no_model_architecture", required=True, view=types.HiddenView()
        )
        return inputs

    model_names = get_model_names(arch)
    model_group = types.RadioGroup()

    for model in model_names:
        model_group.add_choice(model, label=model)

    inputs.enum(
        "model_name",
        model_group.values(),
        label="Model Name",
        description="Select the model name",
        view=types.DropdownView(),
        required=False,
    )

    model_name = ctx.params.get("model_name", None)

    if model_name is None:
        inputs.str("no_model_name", required=True, view=types.HiddenView())
        return inputs

    inputs.view_target(ctx)
    target_view = ctx.target_view()

    label_field_choices = types.AutocompleteView()
    for field in _get_fields_with_type(target_view, fo.Label):
        label_field_choices.add_choice(field, label=field)

    inputs.str(
        "label_field",
        required=True,
        label="Label field",
        description=(
            "The name of a new or existing field in which to store the "
            "predictions"
        ),
        view=label_field_choices,
    )

    inputs.int(
        "batch_size",
        label="Batch size",
        description="Batch size to use when applying the model",
        required=False,
    )

    inputs.int(
        "num_workers",
        label="Number of workers",
        description="Number of workers to use when applying the model",
        required=False,
    )

    inputs.bool(
        "skip_failures",
        label="Skip failures",
        description="Whether to gracefully continue without raising an error if predictions cannot be generated for a sample",
        default=True,
    )

    return inputs


def _apply_transformer_model(ctx):
    target_view = ctx.target_view()

    label_field = ctx.params.get("label_field", None)

    task = ctx.params.get("task")
    zoo_name = task_to_zoo_map[task]
    name_or_path = ctx.params.get("model_name")

    batch_size = ctx.params.get("batch_size", None)
    num_workers = ctx.params.get("num_workers", None)
    skip_failures = ctx.params.get("skip_failures", True)

    delegate = ctx.params.get("delegate", False)

    if not delegate:
        num_workers = 0

    model = foz.load_zoo_model(zoo_name, name_or_path=name_or_path)

    target_view.apply_model(
        model,
        label_field=label_field,
        batch_size=batch_size,
        num_workers=num_workers,
        skip_failures=skip_failures,
    )

    ctx.ops.reload_dataset()


class ApplyTransformerModel(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="apply_transformer_model",
            label="🤗 Apply Transformer model 🤗",
            dynamic=True,
        )
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        _apply_transformer_model_inputs(ctx, inputs)
        _execution_mode(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        _apply_transformer_model(ctx)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)


def register(plugin):
    plugin.register(ApplyTransformerModel)
