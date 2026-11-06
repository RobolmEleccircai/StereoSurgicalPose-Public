"""Paired synchronized video -> predictor callback -> shared surgical overlays."""
import argparse
import importlib
import json
from pathlib import Path
import time
import cv2
from .pipeline import StereoFrame, StereoOverlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('left', 'right', 'calibration', 'instrument-model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--predictor', required=True, help='Python module:factory; factory(config) returns predict(frame, calibration)')
    p.add_argument('--predictor-config', type=Path)
    p.add_argument('--font'); p.add_argument('--show', action='store_true')
    p.add_argument('--geometry-log', action='store_true', help='Save all projected polyline samples; larger per-frame logs')
    p.add_argument('--start-time-ns', type=int, default=0)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()): raise ValueError('Use a new empty output directory')
    calibration = json.loads(a.calibration.read_text()); model = json.loads(a.instrument_model.read_text())
    config = json.loads(a.predictor_config.read_text()) if a.predictor_config else {}
    config['_config_dir'] = str(a.predictor_config.resolve().parent if a.predictor_config else Path.cwd())
    module, factory = a.predictor.rsplit(':', 1)
    predict = getattr(importlib.import_module(module), factory)(config)
    overlay = StereoOverlay(model, calibration, a.font, compact=True)
    for name, value in [('instrument_model', model), ('calibration', calibration)]:
        (a.output/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
    captures = [cv2.VideoCapture(str(path)) for path in (a.left, a.right)]
    writer = None; count = 0; failures = []; sources = set(); timings = []; interrupted = False
    log = (a.output/'frames.jsonl').open('w'); started = time.perf_counter()
    try:
        if not all(cap.isOpened() for cap in captures): raise ValueError('Cannot open paired videos')
        rates = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
        if min(rates) <= 0 or abs(rates[0]-rates[1]) > 1e-6: raise ValueError('Paired video frame rates differ or are invalid')
        fps = rates[0]
        while True:
            items = [cap.read() for cap in captures]
            if items[0][0] != items[1][0]: raise ValueError('Paired videos have different frame counts')
            if not items[0][0]: break
            frame = StereoFrame(f'{count:06d}', a.start_time_ns+round(count*1e9/fps),
                cv2.cvtColor(items[0][1], cv2.COLOR_BGR2RGB), cv2.cvtColor(items[1][1], cv2.COLOR_BGR2RGB))
            start = time.perf_counter()
            try:
                prediction = predict(frame, calibration)
                after_model = time.perf_counter()
                canvas, metadata = overlay.render(frame, prediction)
                metadata['predictor_ms'] = (after_model-start)*1000
                sources.add(prediction.source)
            except (ValueError, KeyError, TypeError) as error:
                reason = str(error); failures.append({'frame_id': frame.frame_id, 'reason': reason})
                canvas = overlay.unavailable(frame, reason)
                metadata = {'frame_id': frame.frame_id, 'timestamp_ns': frame.timestamp_ns, 'error': reason, 'valid': False}
            elapsed = (time.perf_counter()-start)*1000; timings.append(elapsed)
            metadata['predictor_and_overlay_ms'] = elapsed
            if not a.geometry_log: metadata.pop('projection_results', None)
            log.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False)+'\n')
            if writer is None:
                writer = cv2.VideoWriter(str(a.output/'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas.shape[1], canvas.shape[0]))
                if not writer.isOpened(): raise RuntimeError('Video writer unavailable')
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)); count += 1
            if a.show:
                cv2.imshow('Stereo surgical pose', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                wait = max(1, min(1000, round((started+count/fps-time.perf_counter())*1000)))
                if cv2.waitKey(wait) & 0xff in (27, ord('q')):
                    interrupted = True; break
        if count == 0: raise ValueError('Paired videos are empty')
        writer.release(); writer = None
        check = cv2.VideoCapture(str(a.output/'overlay.mp4'))
        valid_video = round(check.get(cv2.CAP_PROP_FRAME_COUNT)) == count and abs(check.get(cv2.CAP_PROP_FPS)-fps) < 1e-6
        check.release()
        if not valid_video: raise ValueError('Encoded video frame count/rate mismatch')
        report = {'completed': not interrupted, 'frames': count, 'input_fps': fps, 'sources': sorted(sources),
            'failed_frames': failures, 'passed': not failures and not interrupted,
            'mean_predictor_and_overlay_ms': sum(timings)/len(timings),
            'note': 'Files must already be synchronized. Frame-index pairing does not establish hardware synchronization.'}
        (a.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report['passed'] else 2
    except BaseException as error:
        (a.output/'report.json').write_text(json.dumps({'passed': False, 'completed': False, 'frames': count, 'error': str(error)}, indent=2)+'\n')
        raise
    finally:
        log.close()
        for cap in captures: cap.release()
        if writer is not None: writer.release()
        if a.show: cv2.destroyAllWindows()


if __name__ == '__main__': raise SystemExit(main())
