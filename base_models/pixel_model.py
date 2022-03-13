import numpy as np
from PIL import Image
from collections import OrderedDict
from model_tools.activations.core import ActivationsExtractorHelper
from model_tools.brain_transformation import ModelCommitment


def get_pixel_model(name):
    model = PixelModel()
    brain_model = ModelCommitment(identifier=name, activations_model=model,
                                  layers=['pixels'])
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    # assert name == 'pixels'
    return brain_model


class PixelModel:
    def __init__(self):
        self._extractor = ActivationsExtractorHelper(identifier='pixels', preprocessing=None,
                                                     get_activations=self._pixels_from_paths)
        self._extractor.insert_attrs(self)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def _pixels_from_paths(self, paths, layer_names):
        np.testing.assert_array_equal(layer_names, ['pixels'])
        pixels = [self._parse_image(path) for path in paths]
        return OrderedDict([('pixels', np.array(pixels))])

    def _parse_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB')  # make sure everything is in RGB and not grayscale L
        image = image.resize((256, 256))  # resize all images to same size
        return np.array(image)


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    # assert name == 'pixels'
    return ['pixels']
