"""Tests for lfx.backends — BackendError, SkyRLBackendInitError, LfXBackend protocol."""

from __future__ import annotations

import pytest

from lfx.backends import BackendError, LfXBackend, SkyRLBackendInitError
from lfx.layers.harness import Harness


# ---------------------------------------------------------------------------
# BackendError — creation and frozen immutability
# ---------------------------------------------------------------------------

class TestBackendError:
    def test_creation(self) -> None:
        err = BackendError(code="gpu_oom", message="Out of memory", recoverable=True)
        assert err.code == "gpu_oom"
        assert err.message == "Out of memory"
        assert err.recoverable is True

    def test_frozen_immutability(self) -> None:
        err = BackendError(code="unknown", message="oops", recoverable=False)
        with pytest.raises((AttributeError, TypeError)):
            err.code = "other"  # type: ignore[misc]

    def test_not_an_exception(self) -> None:
        err = BackendError(code="unknown", message="oops", recoverable=False)
        assert not isinstance(err, Exception)


# ---------------------------------------------------------------------------
# BackendError.from_exception — known type mappings
# ---------------------------------------------------------------------------

class TestBackendErrorFromException:
    def test_memory_error_maps_to_gpu_oom(self) -> None:
        err = BackendError.from_exception(MemoryError("CUDA out of memory"))
        assert err.code == "gpu_oom"
        assert err.recoverable is True

    def test_import_error_maps_to_import_error(self) -> None:
        err = BackendError.from_exception(ImportError("No module named 'vllm'"))
        assert err.code == "import_error"
        assert err.recoverable is False

    def test_module_not_found_maps_to_import_error(self) -> None:
        err = BackendError.from_exception(ModuleNotFoundError("No module named 'skyrl'"))
        assert err.code == "import_error"
        assert err.recoverable is False

    def test_connection_error_maps_to_backend_unreachable(self) -> None:
        err = BackendError.from_exception(ConnectionError("Connection refused"))
        assert err.code == "backend_unreachable"
        assert err.recoverable is True

    def test_timeout_error_maps_to_backend_unreachable(self) -> None:
        err = BackendError.from_exception(TimeoutError("Request timed out"))
        assert err.code == "backend_unreachable"
        assert err.recoverable is True

    def test_type_error_maps_to_schema_incompatible(self) -> None:
        err = BackendError.from_exception(TypeError("Expected dict, got list"))
        assert err.code == "schema_incompatible"
        assert err.recoverable is False

    def test_attribute_error_maps_to_schema_incompatible(self) -> None:
        err = BackendError.from_exception(AttributeError("'NoneType' has no attribute 'x'"))
        assert err.code == "schema_incompatible"
        assert err.recoverable is False


# ---------------------------------------------------------------------------
# BackendError.from_exception — string-based checks
# ---------------------------------------------------------------------------

class TestBackendErrorStringChecks:
    def test_nan_in_message_maps_to_training_diverged(self) -> None:
        err = BackendError.from_exception(RuntimeError("loss is nan, aborting"))
        assert err.code == "training_diverged"
        assert err.recoverable is False

    def test_diverge_in_message_maps_to_training_diverged(self) -> None:
        err = BackendError.from_exception(RuntimeError("training diverged at step 100"))
        assert err.code == "training_diverged"
        assert err.recoverable is False

    def test_config_in_message_maps_to_invalid_config(self) -> None:
        err = BackendError.from_exception(ValueError("bad config: missing field"))
        assert err.code == "invalid_config"
        assert err.recoverable is False

    def test_invalid_in_message_maps_to_invalid_config(self) -> None:
        err = BackendError.from_exception(ValueError("invalid learning rate"))
        assert err.code == "invalid_config"
        assert err.recoverable is False

    def test_unknown_fallback(self) -> None:
        err = BackendError.from_exception(RuntimeError("something weird happened"))
        assert err.code == "unknown"
        assert err.recoverable is False

    def test_message_preserved_from_exception(self) -> None:
        exc = RuntimeError("something weird happened")
        err = BackendError.from_exception(exc)
        assert err.message == str(exc)


# ---------------------------------------------------------------------------
# SkyRLBackendInitError — wraps BackendError correctly
# ---------------------------------------------------------------------------

class TestSkyRLBackendInitError:
    def test_is_exception(self) -> None:
        be = BackendError(code="import_error", message="No vllm", recoverable=False)
        exc = SkyRLBackendInitError(be)
        assert isinstance(exc, Exception)

    def test_wraps_backend_error(self) -> None:
        be = BackendError(code="import_error", message="No vllm", recoverable=False)
        exc = SkyRLBackendInitError(be)
        assert exc.error is be

    def test_str_includes_code_and_message(self) -> None:
        be = BackendError(code="gpu_oom", message="OOM on device 0", recoverable=True)
        exc = SkyRLBackendInitError(be)
        assert "gpu_oom" in str(exc)
        assert "OOM on device 0" in str(exc)

    def test_can_be_raised_and_caught(self) -> None:
        be = BackendError(code="unknown", message="oops", recoverable=False)
        with pytest.raises(SkyRLBackendInitError) as exc_info:
            raise SkyRLBackendInitError(be)
        assert exc_info.value.error is be


# ---------------------------------------------------------------------------
# LfXBackend protocol — Harness satisfies it
# ---------------------------------------------------------------------------

class TestLfXBackendProtocol:
    def test_harness_satisfies_lfx_backend(self) -> None:
        harness = Harness()
        # All required methods must exist
        assert hasattr(harness, "forward_backward")
        assert hasattr(harness, "optim_step")
        assert hasattr(harness, "sample")
        assert hasattr(harness, "save_state")
        assert hasattr(harness, "load_state")
        assert hasattr(harness, "clear_pending_state")
        assert hasattr(harness, "to_dict")

    def test_protocol_runtime_checkable(self) -> None:
        # LfXBackend should be a Protocol that's runtime-checkable
        harness = Harness()
        assert isinstance(harness, LfXBackend)
