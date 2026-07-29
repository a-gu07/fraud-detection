import sys
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from datetime import datetime
import time

from app.models import  Base, engine, ScoredTransactions, reset
from app.scoring import MahalanobisScorer

reset()

streampath = "data/streaming.csv"
histpath = "data/historical.csv"

baseline = pd.read_csv(histpath)
baseline = baseline.loc[baseline.Class == 0]
baseline = baseline.loc[:, ['V' + str(i) for i in range(1, 29, 1)]].to_numpy()

data = pd.read_csv(streampath)
sorted_data = data.sort_values(by='Time').reset_index(drop=True)

features = sorted_data.loc[:, ['V' + str(i) for i in range(1, 29, 1)]].to_numpy()
times = sorted_data.loc[:, 'Time']
labels = sorted_data.loc[:, 'Class']
amounts = sorted_data.loc[:, 'Amount']

global_scorer = MahalanobisScorer.fit(baseline)
adaptive_scorer = MahalanobisScorer.fit_ewma(global_scorer)

prev_time = -1
with Session(engine) as session:
    for x, t, label, amount in zip(features, times, labels, amounts):
        global_score = global_scorer.score(x)
        adaptive_score = adaptive_scorer.score(x)
        del_t = 0 if prev_time == -1 else t - prev_time
        if label == 0:
            adaptive_scorer.update(x, del_t)
            prev_time = t 
        session.add(ScoredTransactions(Time=t, Amount=amount, Class=label, Score=max(global_score, adaptive_score), processed_at=datetime.now()))
        session.commit()
        time.sleep(0.03)
