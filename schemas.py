from pydantic import BaseModel
from typing import List, Optional

class TaskRequest(BaseModel):
    task_id: str

class WriterSample(BaseModel):
    writer_id: str
    folder_path: str

class TaskInfo(BaseModel):
    model_container: str
    model_blob_path: str
    query_container: str
    query_image_path: str
    samples_container: str
    writer_samples: List[WriterSample]

class Prediction(BaseModel):
    writer_id: Optional[str]
    confidence: float

class PredictionResponse(BaseModel):
    task_id: str
    query_image: str
    prediction: Prediction 