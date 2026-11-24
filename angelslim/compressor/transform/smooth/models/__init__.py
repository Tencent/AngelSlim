"""Model-name dispatch for offline Smooth graphs and hook aliases."""


def get_smooth_model(model_name):
    if model_name.lower() == "hy4":
        from . import hy4

        return hy4
    raise ValueError(f"Offline Smooth is not supported for model.name={model_name!r}")


def resolve_model_name(model_name=None, *, model_type=None, profile=None):
    if not model_name:
        for name in ("hy4",):
            adapter = get_smooth_model(name)
            if model_type in adapter.MODEL_TYPES or profile == adapter.LEGACY_PROFILE:
                model_name = name
                break
    name = (model_name or "").lower()
    adapter = get_smooth_model(name)
    if profile not in (None, "offline", adapter.LEGACY_PROFILE):
        raise ValueError(f"Smooth profile {profile!r} does not match model.name={name!r}")
    return name
