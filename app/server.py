"""Brasa API + static frontend.  Run: uvicorn server:app --port 8765  (from app/)"""
import io, json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import engine

app = FastAPI(title="Brasa — harmonized burning activity calendar")
app.add_middleware(GZipMiddleware, minimum_size=2048)
WEB = Path(__file__).parent / "web"


@app.middleware("http")
async def no_cache_assets(request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/assets/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


class Query(BaseModel):
    area: dict
    start: str = "2000-11"
    end: str | None = None
    sensors: list[str] | None = None


def _run(q: Query):
    try:
        return engine.calendar(q.area, q.start, q.end)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/meta")
def meta():
    return engine.meta()


@app.get("/api/globe.bin")
def globe():
    return FileResponse(engine.APP / "globe.bin", media_type="application/octet-stream",
                        headers={"Cache-Control": "max-age=3600"})


@app.get("/api/countries")
def countries():
    return engine.countries()


@app.post("/api/calendar")
def calendar(q: Query):
    sel, clim, yearly, summary = _run(q)
    return {"summary": summary,
            "months": json.loads(sel.to_json(orient="records")),
            "climatology": json.loads(clim.to_json(orient="records")),
            "yearly": json.loads(yearly.to_json(orient="records"))}


def _csv(df, name):
    buf = io.StringIO(); df.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/export/calendar.csv")
def export_calendar(q: Query):
    sel, *_ = _run(q)
    sel = sel.rename(columns={"hotspots": "hotspots_viirs_equivalent", "src": "source",
                              "h_floor": "small_fire_floor", "median": "median_same_month"})
    return _csv(sel, "brasa_calendar.csv")


@app.post("/api/export/points.csv")
def export_points(q: Query):
    end = q.end or engine.meta()["last_month"]
    sensors = tuple(q.sensors or ("modis", "viirs-snpp", "viirs-jpss1"))
    df = engine.points(q.area, q.start, end, sensors)
    return _csv(df, "brasa_hotspots.csv")


if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB), name="assets")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")
