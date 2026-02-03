from __future__ import annotations

from fastapi import FastAPI, HTTPException

from turbine_asset_health.app.schemas import AnomalyRequest, AnomalyResponse, BatchAnomalyRequest
from turbine_asset_health.models.infer import score_one

app = FastAPI(title="Wind Turbine Asset Health — Anomaly API", version="0.1.0")

def _dump(m):
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/anomaly", response_model=AnomalyResponse)
def anomaly(req: AnomalyRequest):
    try:
        result = score_one(_dump(req))
        return AnomalyResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")

@app.post("/batch_anomaly")
def batch_anomaly(req: BatchAnomalyRequest):
    try:
        out = [score_one(_dump(row)) for row in req.rows]
        return {"results": out, "count": len(out)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")
