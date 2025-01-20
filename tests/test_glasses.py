from src.glasses.common import is_glasses_connected
import pytest


@pytest.mark.asyncio
async def test_glasses_disconnected() -> None:
    is_connected = await is_glasses_connected()
    assert not is_connected
