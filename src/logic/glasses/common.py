import asyncio

from g3pylib import connect_to_glasses
from src.core.config import DEFAULT_GLASSES_HOSTNAME


async def is_connected(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> bool:
    try:
        async def connect():
            async with connect_to_glasses.with_hostname(glasses_hostname, using_ip=True):
                return True

        # Use asyncio.wait_for to apply a timeout
        return await asyncio.wait_for(connect(), timeout=1)  # Timeout after 2 seconds
    except asyncio.TimeoutError:
        return False

async def get_battery_level(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> float:
    try:
        async with connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3:
            return round((await g3.system.battery.get_level()) * 100)
    except asyncio.TimeoutError:
        return -1
