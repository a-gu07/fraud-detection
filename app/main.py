from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.models import get_db, ScoredTransactions
from pydantic import BaseModel
from datetime import datetime
from fastapi import HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


import numpy as np
import pandas as pd

from app.scoring import MahalanobisScorer

global_scorer: MahalanobisScorer = None
adaptive_scorer: MahalanobisScorer = None
prev_time = -1

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_scorer, adaptive_scorer

    baseline = pd.read_csv("data/historical.csv")
    baseline = baseline.loc[baseline.Class == 0]
    baseline = baseline.loc[:, ['V' + str(i) for i in range(1, 29, 1)]].to_numpy()

    global_scorer = MahalanobisScorer.fit(baseline)
    adaptive_scorer = MahalanobisScorer.fit_ewma(global_scorer)
    yield

origins = ['*']

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=['GET', 'POST'],
    allow_headers=['*']
)

class Transaction(BaseModel):
    id: int
    Time: float
    Amount: float
    Class: int
    Score: float
    processed_at: datetime

    model_config = {'from_attributes': True}

class IncomingTransaction(BaseModel):
    Time: float
    Amount: float
    Class: int
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get('/')
def root_route():
    return {'message': 'working!'}

@app.get('/health')
def health(db: Session = Depends(get_db)):
    try: 
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail=f"Error: Database is unreachable")
    return {"status": "ok", "db": "ok"} 
    
@app.get("/transactions", response_model=list[Transaction])
def list_transactions(
    limit: int = 100,
    offset: int = 0,
    after_id: int = 0,
    db: Session = Depends(get_db),
):  
    if after_id > 0:
        stmt = text("SELECT * FROM scored_transactions WHERE id > :after_id ORDER BY id ASC")
        result = db.execute(stmt, {'after_id': after_id})
        return result.all()
    stmt = text("SELECT * FROM scored_transactions ORDER BY processed_at DESC LIMIT :limit OFFSET :offset")
    result = db.execute(stmt, {'limit': limit, 'offset': offset})
    return result.all()

@app.get("/alerts", response_model=list[Transaction])
def list_alerts(
    limit: int = 100,
    offset: int = 0,
    score: float = 19.08, # score calculated from notebook 5
    db: Session = Depends(get_db),
):
    stmt = text("SELECT * FROM scored_transactions WHERE SCORE >= :score ORDER BY processed_at DESC LIMIT :limit OFFSET :offset")
    result = db.execute(stmt, {'limit': limit, 'offset': offset, 'score': score})
    return result.all()

@app.get("/stats")
def list_stats(
    threshold: float = 19.08, # score calculated from notebook 5
    db: Session = Depends(get_db),
):
    stmt = ("SELECT COUNT(*) AS Total, "
    "SUM(CASE WHEN Score >= :threshold THEN 1 ELSE 0 END) as Alert, "
    "SUM(CASE WHEN Score >= :threshold AND Class = 1 THEN 1 ELSE 0 END) as true_positives,"
    "SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as actual_positives,"
    "SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) as actual_negatives"
    " FROM scored_transactions"
    )
    result = db.execute(text(stmt), {'threshold': threshold}).one()
    total = result.Total if result.Total is not None else 0
    alerts = result.Alert if result.Alert is not None else 0
    tp = result.true_positives if result.true_positives is not None else 0
    ap = result.actual_positives if result.actual_positives is not None else 0
    an = result.actual_negatives if result.actual_negatives is not None else 0

    precision = tp / alerts if alerts != 0 else None
    recall = tp / ap if ap != 0 else None
    fpr = (alerts - tp) / an if an != 0 else None

    return {'total': total, 'alerts': alerts,'threshold': threshold, 'precision': precision, 'recall': recall, 'fpr': fpr}

@app.post("/transactions", response_model=Transaction)
def submit_transaction(transaction: IncomingTransaction, db: Session = Depends(get_db)):
    global prev_time

    if transaction.Class == 0 and prev_time != -1 and transaction.Time <= prev_time:
        raise HTTPException(
            status_code=400,
            detail=f"Time {transaction.Time} is not after the last recorded update time {prev_time}."
        )
    
    x = np.array([getattr(transaction, f'V{i}') for i in range(1, 29)])

    global_score = global_scorer.score(x)
    adaptive_score = adaptive_scorer.score(x)
    combined = max(global_score, adaptive_score)

    row = ScoredTransactions(
        Time=transaction.Time,
        Amount=transaction.Amount,
        Class=transaction.Class,
        Score=combined,
        processed_at=datetime.now(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    del_t = 0 if prev_time == -1 else transaction.Time - prev_time
    if transaction.Class == 0:
        adaptive_scorer.update(x, del_t)
        prev_time = transaction.Time

    return row