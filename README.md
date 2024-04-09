# Hugging Face Plugins for FiftyOne 🤗 ➕ <img src="https://gist.githubusercontent.com/jacobmarks/eb18cc90596f7310e4dad1be2526c070/raw/e05e51be697a9501f64fe8d1b7008fc5ebe56369/fiftyone_icon.svg" style="vertical-align: -5px;" width="30" height="30" alt="fiftyone icon">

This repository contains a collection of plugins for the [FiftyOne](https://fiftyone.ai/), which make it easier than ever to connect your models to your data.

In particular, the plugins in this repository make the functionality from the FiftyOne SDK's [Hugging Face integration](https://docs.voxel51.com/integrations/huggingface.html) available directly within the FiftyOne App.

The plugins in this repository are:

- [`@voxel51/huggingface_hub`](./plugins/huggingface_hub): A plugin that allows you to push FiftyOne datasets to the Hugging Face Hub, and load datasets from the Hub into FiftyOne.
- [`@voxel51/transformers`](./plugins/transformers): A plugin that allows you to run inference on your datasets using Hugging Face Transformers models.

## Installation

First, install the latest version of FiftyOne and Hugging Face libraries:

```bash
pip install -U fiftyone transformers huggingface_hub
```

Then, download the plugins:

```bash
fiftyone plugins download https://github.com/voxel51/fiftyone_huggingface_plugins
```

To install just one of the plugins, you can specify the plugin name. For example, to install the `@voxel51/transformers` plugin:

```bash
fiftyone plugins download https://github.com/voxel51/fiftyone_huggingface_plugins --plugin-names @voxel51/transfomers
```