import json
from pathlib import Path

import aiohttp
from fastapi import Request


async def download_file(url: str, target_path: Path) -> None:
    """
    Downloads a file from a URL and saves it to a local path asynchronously.

    Args:
        url (str): The URL of the file to download.
        local_path (str): The local path where the file will be saved.

    Returns:
        None

    Raises:
        aiohttp.ClientError: If there is an error during the HTTP request.
        IOError: If there is an error writing the file to disk.
    """
    with target_path.open("wb") as fd:
        async with aiohttp.ClientSession() as session, session.get(url) as resp:
            async for chunk in resp.content.iter_chunked(1024 * 64):
                fd.write(chunk)


def save_json(data: dict[str, str], target_path: Path) -> None:
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json_files(target_path: Path) -> list[dict[str, str]]:
    files = [file for file in target_path.iterdir() if file.suffix == ".json"]
    return [json.load(file.open(encoding="utf-8")) for file in files]


def is_hx_request(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"
