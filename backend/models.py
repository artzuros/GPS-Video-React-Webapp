from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    file_path = Column(String)
    duration = Column(Float)

    gps_points = relationship("GPSPoint", back_populates="video")


class GPSPoint(Base):
    __tablename__ = "gps_points"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    lat = Column(Float)
    lon = Column(Float)
    highlight = Column(Boolean)
    timestamp = Column(Float)

    video = relationship("Video", back_populates="gps_points")

class InferenceResult(Base):
    __tablename__ = "inference_ing"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"))
    inference_results_path = Column(String, nullable=False)
    heatmap_path = Column(String, nullable=True)
    created_at = Column(String)

    video = relationship("Video")
