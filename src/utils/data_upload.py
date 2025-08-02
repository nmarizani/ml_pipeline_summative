import os
import shutil
import pandas as pd  # Only needed if you plan to use META_FILE
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi import UploadFile  # <-- MISSING IMPORT!
from src.db import UploadedImage

UPLOAD_DIR = "uploads/"
META_FILE = os.path.join(UPLOAD_DIR, "metadata.csv")
RETRAIN_THRESHOLD = 1


def save_bulk_images(file_paths_labels, db: Session):
    """
    Save multiple images and their metadata to the database.

    Args:
        file_paths_labels (list): List of (file_path, label) tuples.
        db (Session): SQLAlchemy DB session.

    Returns:
        int: Number of entries added.
    """
    entries = []

    for file_path, label in file_paths_labels:
        filename = os.path.basename(file_path)
        entries.append(
            UploadedImage(
                filename=filename,
                label=label,
            )
        )

    db.bulk_save_objects(entries)
    db.commit()

    return len(entries)


def should_trigger_retraining(db: Session) -> bool:
    """
    Check if the retraining threshold has been met.

    Args:
        db (Session): SQLAlchemy DB session.

    Returns:
        bool: True if retraining should be triggered.
    """
    count = db.query(UploadedImage).count()
    return count > 0 and count % RETRAIN_THRESHOLD == 0


def save_uploaded_image(file: UploadFile, upload_dir="data/uploads") -> str:
    """
    Save a single uploaded file to disk.

    Args:
        file (UploadFile): File to save.
        upload_dir (str): Directory to store files.

    Returns:
        str: Path where file was saved.
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path