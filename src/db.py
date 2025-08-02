from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, func

SQLALCHEMY_DATABASE_URL = "sqlite:///./vaccine_data.db"  # local file

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class UploadedImage(Base):
    __tablename__ = "uploaded_images"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=False)

class RetrainHistory(Base):
    __tablename__ = "retrain_history"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, nullable=False)
    notes = Column(String)