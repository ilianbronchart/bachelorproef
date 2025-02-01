
from dataclasses import dataclass
from fastapi import Request
from fastapi.responses import HTMLResponse

import src.logic.glasses as glasses
from src.api import inference, labeling, recordings
from src.api.settings import Template, app, templates, BaseContext

app.include_router(recordings.router)
app.include_router(inference.router)
app.include_router(labeling.router)

@dataclass
class GlassesConnectionContext(BaseContext):
    glasses_connected: bool
    battery_level: int

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(Template.INDEX, {"request": request})

@app.get("/glasses/connection", response_class=HTMLResponse)
async def glasses_connection(request: Request):
    """Retrieve connection details for the glasses"""
    context = GlassesConnectionContext(
        request=request,
        glasses_connected=await glasses.is_connected(), 
        battery_level=await glasses.get_battery_level()
    )
    return templates.TemplateResponse(Template.CONNECTION_STATUS, context.to_dict())