import importlib
import inspect
import pkgutil

from ..utils.config_manager import ConfigManager
from .base import BaseModel

EXCLUDED_CLASSES = [
    'BaseModel',
]


def get_subclasses(base_class, package_name):
    """Get all subclasses of base_class in the specified package"""
    all_classes = []

    package = importlib.import_module(package_name)

    for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
        if not name.startswith('__'):
            try:
                module = importlib.import_module(f'{package_name}.{name}')
                classes = [
                    cls for _, cls in inspect.getmembers(module, inspect.isclass)
                    if issubclass(cls, base_class) and cls.__name__ not in EXCLUDED_CLASSES
                ]
                all_classes.extend(classes)
            except Exception as e:
                print(f'Warning: Failed to load module {name}: {str(e)}')

    return all_classes


supported_models = get_subclasses(BaseModel, __package__)
supported_models = {model.NAME: model for model in supported_models}


def build_model(name, **kwargs):
    if name not in supported_models:
        raise ValueError(
            f'Model {name} not supported, all supported models: {supported_models.keys()}')
    return supported_models[name](**kwargs)
