import asyncio

from g3pylib import connect_to_glasses
from src.core.config import DEFAULT_GLASSES_HOSTNAME


async def is_glasses_connected(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> bool:
    try:
        async with connect_to_glasses.with_hostname(glasses_hostname, using_ip=True):
            return True
    except asyncio.TimeoutError:
        return False
