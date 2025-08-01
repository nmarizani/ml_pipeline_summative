from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "postgresql://username:password@localhost/vaccine_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class UploadedImage(Base):
    __tablename__ = "uploaded_images"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    label = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class RetrainHistory(Base):
    __tablename__ = "retrain_history"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    notes = Column(String)