"""Replay saved RGB and 17 truth-derived values; does not render or run a model."""
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image
from .adapters import load_pair
from .pipeline import StereoOverlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'instrument-model', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--font'); a = p.parse_args()
    if a.output.resolve().is_relative_to(a.run.resolve()): raise ValueError('Keep output outside the source dataset')
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()): raise ValueError('Use a new empty output directory')
    model = json.loads(a.instrument_model.read_text()); overlay = None; writers = {}; count = 0; rows = []
    try:
        for path in sorted(a.run.glob('sequences/*/labels/*.json')):
            if 'eyes' not in json.loads(path.read_text()): continue
            frame, prediction, calibration, label = load_pair(path)
            prediction.visibility = {eye: e['fixed_tip_visibility'] for eye, e in label['eyes'].items()}
            prediction.visibility_source = 'simulation_ground_truth'
            if overlay is None:
                overlay = StereoOverlay(model, calibration, a.font, compact=True)
                report_overlay = StereoOverlay(model, calibration, a.font, compact=False)
            elif overlay.cameras != calibration:
                raise ValueError('Calibration changed during replay; instantiate the matching camera configuration')
            quality = {eye: {k: e[k] for k in ('jaw_tip_region_visibility', 'image_quality')} for eye, e in label['eyes'].items()}
            canvas, metadata = overlay.render(frame, prediction)
            full, _ = report_overlay.render(frame, prediction, quality)
            stem = label['sequence_id']+'_'+label['job_id']
            Image.fromarray(canvas).save(a.output/(stem+'_video.jpg'), quality=95, subsampling=0)
            Image.fromarray(full).save(a.output/(stem+'_report.jpg'), quality=95, subsampling=0)
            (a.output/(stem+'.json')).write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
            sid = label['sequence_id']
            if sid not in writers:
                writer = cv2.VideoWriter(str(a.output/(sid+'_60fps.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), 60, (canvas.shape[1], canvas.shape[0]))
                if not writer.isOpened(): raise RuntimeError('Cannot create replay video')
                writers[sid] = writer
            writers[sid].write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)); count += 1
            rows.append({'sequence_id': sid, 'frame_index': label['frame_index'], 'timestamp_ns': frame.timestamp_ns,
                         'report_preview': stem+'_report.jpg', 'video_preview': stem+'_video.jpg'})
        if count == 0: raise ValueError('No labeled frames')
        for writer in writers.values(): writer.release()
        for sid in writers:
            frames = [r for r in rows if r['sequence_id'] == sid]
            if any(b['frame_index'] != a['frame_index']+1 or abs((b['timestamp_ns']-a['timestamp_ns'])-1e9/60)>1
                   for a, b in zip(frames, frames[1:])): raise ValueError('Replay requires consecutive frames per sequence')
            cap = cv2.VideoCapture(str(a.output/(sid+'_60fps.mp4')))
            ok = round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames) and abs(cap.get(cv2.CAP_PROP_FPS)-60)<1e-6
            cap.release()
            if not ok: raise ValueError('Replay video count/rate mismatch')
        (a.output/'report.json').write_text(json.dumps({'passed': True, 'pairs': count, 'source': 'ground_truth_replay',
            'new_rgb_renders': 0, 'trained_model_used': False, 'frames': rows}, ensure_ascii=False, indent=2)+'\n')
        print('REPLAY_COMPLETE', count)
    except BaseException as error:
        (a.output/'report.json').write_text(json.dumps({'passed': False, 'pairs': count, 'error': str(error)}, indent=2)+'\n')
        raise
    finally:
        for writer in writers.values(): writer.release()


if __name__ == '__main__': main()
