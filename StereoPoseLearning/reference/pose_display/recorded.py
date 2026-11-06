"""Video integration fixture: recorded truth-derived values, NEVER a model."""
import json
from pathlib import Path
from .pipeline import PosePrediction


def build_predictor(config):
    path = Path(config['_config_dir'])/config['predictions_json']
    rows = json.loads(path.read_text())['records']
    records = {r['frame_id']: r for r in rows}
    if len(records) != len(rows): raise ValueError('Duplicate recorded prediction frame IDs')

    def predict(frame, calibration):
        row = records[frame.frame_id]
        return PosePrediction(row['frame_id'], row['timestamp_ns'], row['values17'], source='ground_truth_replay')
    return predict
