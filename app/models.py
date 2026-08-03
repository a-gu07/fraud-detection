import os
from sqlalchemy import create_engine, text
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, Session, sessionmaker

class Base(DeclarativeBase):
    pass


class ScoredTransactions(Base):
    __tablename__ = "scored_transactions"

    id: Mapped[int] = mapped_column('id', primary_key=True)
    Time: Mapped[float] = mapped_column('time')
    Amount: Mapped[float] = mapped_column('amount')
    Class: Mapped[int] = mapped_column('class')
    Score: Mapped[float] = mapped_column('score')
    processed_at: Mapped[datetime] = mapped_column('processed_at')


def reset():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data/fraud_detection.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
if engine.dialect.name == 'sqlite':
    with engine.connect() as conn:
        conn.execute(text('PRAGMA journal_mode=WAL'))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
