import pytest

import src.api.controllers.glasses_controller as glasses


@pytest.mark.asyncio
async def test_glasses_disconnected() -> None:
    is_connected = await glasses.is_connected()
    assert not is_connected
