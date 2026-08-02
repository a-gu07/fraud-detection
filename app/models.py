from sqlalchemy import create_engine, text
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, Session, sessionmaker

class Base(DeclarativeBase):
    pass


class ScoredTransactions(Base):
    __tablename__ = "scored_transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    Time: Mapped[float] = mapped_column()
    Amount: Mapped[float] = mapped_column()
    Class: Mapped[int] = mapped_column()
    Score: Mapped[float] = mapped_column()
    processed_at: Mapped[datetime] = mapped_column()


def reset():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

engine = create_engine("sqlite:///data/fraud_detection.db")
SessionLocal = sessionmaker(bind=engine)
with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
