"""Hugging Face Hub plugin.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types


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


def get_fiftyone_hub_datasets():
    # pylint: disable=no-name-in-module,import-error
    from huggingface_hub import HfApi

    api = HfApi()
    gen = api.list_datasets(filter="fiftyone")
    dataset_ids = [dataset.id for dataset in gen]
    return sorted(dataset_ids)


class PushToHuggingFaceHub(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="push_to_huggingface_hub",
            label="🤗 Push dataset to Hugging Face Hub 🤗",
            dynamic=True,
        )
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        _push_to_hub_inputs(inputs, ctx)
        return types.Property(inputs)

    def execute(self, ctx):
        _push_to_hub_execute(ctx)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)


class LoadFromHuggingFaceHub(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="load_from_huggingface_hub",
            label="🤗 Load dataset from Hugging Face Hub 🤗",
            allow_delegated_execution=True,
            dynamic=True,
        )
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        _load_from_hub_inputs(inputs, ctx)
        return types.Property(inputs)

    def resolve_output(self, ctx):
        pass

    def execute(self, ctx):
        _load_from_hub_execute(ctx)


class HuggingFaceHub(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="huggingface_hub",
            label="Hugging Face Hub",
            unlisted=True,
            dynamic=True,
        )
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        _hf_hub_choice_input(inputs, ctx)

        if ctx.params.get("hf_action") == "Push to Hugging Face Hub":
            _push_to_hub_inputs(inputs, ctx)
        else:
            _load_from_hub_inputs(inputs, ctx)
        return types.Property(inputs)

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="🤗",
                description="Load dataset from Hugging Face Hub",
            ),
        )

    def execute(self, ctx):
        if ctx.params.get("hf_action") == "Push to Hugging Face Hub":
            _push_to_hub_execute(ctx)
        else:
            _load_from_hub_execute(ctx)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)


def _hf_hub_choice_input(inputs, ctx):
    choices = ["Push to Hugging Face Hub", "Load from Hugging Face Hub"]

    group = types.RadioGroup()

    for choice in choices:
        group.add_choice(choice, label=choice)

    inputs.enum(
        "hf_action",
        group.values(),
        label="Push or Load?",
        view=types.TabsView(),
        required=True,
    )


def _get_repo_id(repo_name):
    # pylint: disable=no-name-in-module,import-error
    from huggingface_hub import whoami

    hf_username = whoami()["name"]
    try:
        repo_id = hf_username + "/" + repo_name
    except:
        repo_id = hf_username + "/" + "<repo_name>"
    return repo_id


def _repo_name_input(inputs, ctx):
    repo_name = ctx.params.get("repo_name", None)
    repo_id = _get_repo_id(repo_name)

    inputs.str(
        "repo_name",
        label="Repo Name",
        description=(
            "Repo name to assign to the dataset. Dataset will be push to "
            "Hugging Face Hub under the repo ID: %s" % repo_id
        ),
        required=True,
    )


def _license_input(inputs):
    licenses_group = types.RadioGroup()

    for license in LICENSES:
        licenses_group.add_choice(license, label=license)

    inputs.enum(
        "license",
        licenses_group.values(),
        label="License",
        description="License to assign to the dataset.",
        view=types.AutocompleteView(),
        required=False,
    )


def _push_to_hub_inputs(inputs, ctx):
    _repo_name_input(inputs, ctx)
    _license_input(inputs)

    inputs.str(
        "description",
        label="Description",
        description="Optional description of the dataset",
        required=False,
    )

    inputs.int(
        "chunk_size",
        label="Chunk size",
        description="Number of media files to upload in a single directory",
        required=False,
    )

    inputs.str(
        "tags",
        label="Tags",
        description="Comma-separated list of tags to assign to the dataset",
        required=False,
    )

    inputs.bool(
        "private",
        label="Private",
        description="Whether the dataset should be private",
        default=False,
    )

    inputs.bool(
        "exist_ok",
        label="Exist OK",
        description="Whether to overwrite an existing dataset",
        default=False,
    )

    inputs.view_target(ctx)
    _execution_mode(ctx, inputs)

    file_explorer = types.FileExplorerView(button_label="Choose a file...")
    prop = inputs.file(
        "preview_path",
        required=True,
        label="Preview Image",
        description="Choose a preview image for the dataset card",
        view=file_explorer,
    )

    preview_filepath = _parse_path(ctx, "preview_path")
    if preview_filepath is None:
        return False

    import os

    if os.path.splitext(preview_filepath)[1] not in PREVIEW_EXTS:
        prop.invalid = True
        prop.error_message = f"Please provide a file with one of the following extensions: {PREVIEW_EXTS}"
        return False


def _parse_path(ctx, key):
    value = ctx.params.get(key, None)
    return value.get("absolute_path", None) if value else None


def _push_to_hub_execute(ctx):
    sample_collection = ctx.target_view()

    repo_name = ctx.params.get("repo_name")
    description = ctx.params.get("description", None)
    license = ctx.params.get("license", None)
    tags = ctx.params.get("tags", None)
    private = ctx.params.get("private", False)
    exist_ok = ctx.params.get("exist_ok", False)
    chunk_size = ctx.params.get("chunk_size", None)
    preview_filepath = _parse_path(ctx, "preview_path")

    if isinstance(tags, str):
        tags = tags.split(",")
        tags = [tag.strip() for tag in tags]

    from fiftyone.utils.huggingface import push_to_hub

    push_to_hub(
        sample_collection,
        repo_name,
        description=description,
        license=license,
        tags=tags,
        private=private,
        exist_ok=exist_ok,
        chunk_size=chunk_size,
        preview_path=preview_filepath,
    )


def _repo_id_input(inputs):
    dataset_choices = get_fiftyone_hub_datasets()

    datasets_group = types.RadioGroup()

    for choice in dataset_choices:
        datasets_group.add_choice(choice, label=choice)

    inputs.enum(
        "repo_id",
        datasets_group.values(),
        label="Dataset",
        description="Repo ID of dataset to load from Hugging Face Hub",
        view=types.AutocompleteView(),
        required=False,
    )


def _load_from_hub_inputs(inputs, ctx):
    _repo_id_input(inputs)

    inputs.int(
        "max_samples",
        label="Max samples",
        description="Maximum number of samples to load",
    )
    inputs.int(
        "batch_size",
        label="Batch size",
        description="Batch size to use when loading samples",
    )
    inputs.str(
        "name",
        label="Name",
        description=(
            "Name to assign to the loaded dataset. If not provided,"
            " the name of the dataset on Hugging Face Hub will be used"
        ),
        required=False,
    )

    inputs.bool(
        "persistent",
        label="Persistent",
        description="Whether to persist the dataset to disk",
        default=False,
    )
    inputs.bool(
        "overwrite",
        label="Overwrite",
        description="Whether to overwrite an existing dataset",
        default=False,
    )
    inputs.int(
        "num_workers",
        label="Number of workers",
        description=(
            "Number of workers to use when downloading media. If not "
            "delegated, this will be set to 1"
        ),
    )
    inputs.bool(
        "open_dataset",
        label="Open dataset",
        description="Whether to open the dataset after loading",
        default=True,
    )
    _execution_mode(ctx, inputs)


def _load_from_hub_execute(ctx):
    repo_id = ctx.params.get("repo_id")
    max_samples = ctx.params.get("max_samples", None)
    batch_size = ctx.params.get("batch_size", None)
    name = ctx.params.get("name", None)
    persistent = ctx.params.get("persistent", False)
    overwrite = ctx.params.get("overwrite", False)
    delegate = ctx.params.get("delegate", False)
    num_workers = 1 if not delegate else ctx.params.get("num_workers", None)

    open_dataset = ctx.params.get("open_dataset", True)

    from fiftyone.utils.huggingface import load_from_hub

    dataset = load_from_hub(
        repo_id,
        max_samples=max_samples,
        batch_size=batch_size,
        name=name,
        persistent=persistent,
        overwrite=overwrite,
        num_workers=num_workers,
    )

    if open_dataset:
        ctx.ops.open_dataset(dataset.name)


def register(plugin):
    plugin.register(PushToHuggingFaceHub)
    plugin.register(LoadFromHuggingFaceHub)
    plugin.register(HuggingFaceHub)
