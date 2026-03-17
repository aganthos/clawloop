"""Tests for run_async sync/async bridge utility."""

import asyncio

import pytest

from lfx.utils.async_bridge import run_async


async def _return_value(val):
    return val


async def _raise_error(msg):
    raise ValueError(msg)


async def _concurrent_sum(n: int) -> int:
    results = await asyncio.gather(*[_return_value(i) for i in range(n)])
    return sum(results)


class TestRunAsyncBasic:
    def test_returns_int(self) -> None:
        result = run_async(_return_value(42))
        assert result == 42

    def test_returns_string(self) -> None:
        result = run_async(_return_value("hello"))
        assert result == "hello"

    def test_returns_list(self) -> None:
        result = run_async(_return_value([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_returns_none(self) -> None:
        result = run_async(_return_value(None))
        assert result is None


class TestRunAsyncExceptions:
    def test_propagates_exception(self) -> None:
        with pytest.raises(ValueError, match="something went wrong"):
            run_async(_raise_error("something went wrong"))

    def test_propagates_exception_type(self) -> None:
        async def raise_type_error():
            raise TypeError("bad type")

        with pytest.raises(TypeError):
            run_async(raise_type_error())


class TestRunAsyncGather:
    def test_gather_concurrent_coroutines(self) -> None:
        result = run_async(_concurrent_sum(5))
        assert result == 10  # 0+1+2+3+4

    def test_gather_returns_list(self) -> None:
        async def gather_values():
            return await asyncio.gather(
                _return_value("a"),
                _return_value("b"),
                _return_value("c"),
            )

        result = run_async(gather_values())
        assert result == ["a", "b", "c"]


class TestRunAsyncFromRunningLoop:
    def test_works_from_running_loop(self) -> None:
        """run_async called from within a running event loop uses thread executor."""

        async def caller():
            # This simulates calling run_async from within an async context
            # (e.g. Jupyter, async orchestration)
            return run_async(_return_value(99))

        result = asyncio.run(caller())
        assert result == 99

    def test_exception_from_running_loop(self) -> None:
        """Exceptions propagate correctly when called from a running loop."""

        async def caller():
            return run_async(_raise_error("nested error"))

        with pytest.raises(ValueError, match="nested error"):
            asyncio.run(caller())
