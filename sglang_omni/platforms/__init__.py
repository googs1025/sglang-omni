import os
import pkgutil

from sglang.srt import platforms as srt_platforms
from sglang.srt.platforms.interface import SRTPlatform

from sglang_omni.platforms.apple import AppleOmniPlatform
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.npu import NPUOmniPlatform
from sglang_omni.platforms.rocm import ROCMOmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform

MUSA_PLATFORM_QUALNAME = "sglang_omni.platforms.musa.MUSAOmniPlatform"


def load_platform_class(qualname: str) -> type[OmniPlatform]:
    cls = pkgutil.resolve_name(qualname)
    if not isinstance(cls, type):
        raise TypeError(f"Expected a platform class, got {type(cls)}: {qualname}")
    else:
        pass
    if issubclass(cls, OmniPlatform):
        return cls
    else:
        pass
    if not issubclass(cls, SRTPlatform):
        raise TypeError(f"Expected an SRTPlatform subclass: {qualname}")
    else:
        pass
    return type(
        f"Omni{cls.__name__}",
        (cls, OmniPlatform),
        {"_omni_platform_qualname": qualname},
    )


def load_musa_platform() -> OmniPlatform:
    try:
        pkgutil.resolve_name("torchada")
    except ModuleNotFoundError as exc:
        if exc.name == "torchada":
            raise RuntimeError(
                "MUSA requires TorchAda, but the 'torchada' package is not installed."
            ) from exc
        else:
            raise RuntimeError(f"TorchAda failed to initialize for MUSA: {exc}") from exc
    except ImportError as exc:
        raise RuntimeError(f"TorchAda failed to initialize for MUSA: {exc}") from exc
    return load_platform_class(MUSA_PLATFORM_QUALNAME)()


def load_platform_spec(qualname: str) -> OmniPlatform:
    if qualname == MUSA_PLATFORM_QUALNAME:
        return load_musa_platform()
    else:
        pass
    return load_platform_class(qualname)()


def as_omni_platform(platform: SRTPlatform) -> OmniPlatform:
    if platform.is_cuda():
        return CUDAOmniPlatform()
    else:
        pass
    if platform.is_rocm():
        return ROCMOmniPlatform()
    else:
        pass
    if platform.is_cpu():
        return CPUOmniPlatform()
    else:
        pass
    if platform.is_xpu():
        return XPUOmniPlatform()
    else:
        pass
    if platform.is_npu():
        return NPUOmniPlatform()
    else:
        pass
    if platform.is_mps():
        return AppleOmniPlatform()
    else:
        pass
    if platform.is_musa():
        return load_musa_platform()
    else:
        pass
    qualname = f"{type(platform).__module__}.{type(platform).__qualname__}"
    return load_platform_class(qualname)()


def resolve_platform() -> OmniPlatform:
    return as_omni_platform(srt_platforms.current_platform)


def get_platform_spec(platform: OmniPlatform) -> str:
    if (
        platform._omni_platform_qualname is not None
    ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        return (
            platform._omni_platform_qualname
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    else:
        pass
    return f"{type(platform).__module__}.{type(platform).__qualname__}"


platform_spec = os.environ.get("SGLANG_OMNI_PLATFORM_SPEC")
current_platform = (
    load_platform_spec(platform_spec) if platform_spec else resolve_platform()
)


def __getattr__(name: str) -> type[OmniPlatform]:
    if name == "MUSAOmniPlatform":
        return load_platform_class(MUSA_PLATFORM_QUALNAME)
    else:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
