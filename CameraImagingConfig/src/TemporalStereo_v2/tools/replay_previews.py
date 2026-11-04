"""Exercise the capture preview publisher with saved data; no simulator imports."""
import argparse
import copy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from io_utils import read,write,sha
from live_preview import LivePreview
from pose_truth import attach_body_truth,validate_body_truth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--font',type=Path,required=True)
    p.add_argument('--compare',type=Path,help='Optional previously approved previews for byte equality')
    a=p.parse_args()
    if a.output.resolve().is_relative_to((a.run/'sequences').resolve()):
        raise ValueError('Preview output cannot be inside raw sequences')
    labels=sorted(a.run.glob('sequences/*/labels/*.json'))
    if not labels:raise ValueError('No saved labels')
    protected=[p for p in (a.run/'sequences').rglob('*') if p.is_file()]
    protected += [p for p in (a.run/'pose_geometry').glob('*') if p.is_file()]
    before={str(p):sha(p) for p in protected}
    old_geometry=read(a.run/'pose_geometry/geometry.json') if any('body_pose_truth' not in read(p) for p in labels) else None
    if old_geometry:
        hinges=read(a.run/'pose_geometry/jaw_hinges.json')
        if hinges['pose_geometry_sha256']!=sha(a.run/'pose_geometry/geometry.json'):
            raise ValueError('Legacy hinge provenance mismatch')
        old_frames={(f['sequence_id'],f['job_id']):f for f in old_geometry['frames']}
    publisher=LivePreview(a.output,a.font);count=0
    for label in labels:
        record=read(label);truth_path=label.parent/record['camera_truth']
        if sha(truth_path)!=record['camera_truth_sha256']:raise ValueError('Camera truth hash mismatch')
        truth=read(truth_path)
        if 'body_pose_truth' not in record:
            frame=old_frames[(record['sequence_id'],record['job_id'])]
            if frame['label_sha256']!=sha(label):raise ValueError('Legacy pose truth mismatch')
            attach_body_truth(record,frame['T_world_from_body_mm'],frame['roll_body'],
                              {'unit':'mm','jaws':hinges['jaws']},truth,old_geometry['source_scene_sha256'])
        images={}
        for eye,e in record['eyes'].items():
            path=label.parent/e['image']
            if sha(path)!=e['image_sha256']:raise ValueError('Image hash mismatch')
            with Image.open(path) as image:images[eye]=image.convert('RGB')
        target=publisher.publish(record,truth,images,sha(label));count+=1
        if a.compare and sha(target)!=sha(a.compare/target.name):
            raise ValueError('Capture display differs from approved offline display')
    bad=copy.deepcopy(record)
    bad['eyes']['left']['local_frames']['shaft']['T_optical_from_local_mm'][0][0]+=.1
    try:validate_body_truth(bad,truth)
    except ValueError:pass
    else:raise AssertionError('Corrupt rotation was accepted')
    # An interrupted preview can be repaired without rendering or modifying labels.
    (a.output/'latest.jpg').unlink()
    publisher.publish(record,truth,images,sha(label))
    assert sha(a.output/'latest.jpg')==sha(target)
    assert before=={str(p):sha(p) for p in protected}
    report={'passed':True,'pairs_replayed':count,'new_rgb_renders':0,
            'original_files_unchanged':len(before),'corrupt_rotation_rejected':True,
            'interrupted_preview_repaired':True,'full_pose_frames_per_eye':['shaft','jaw1','jaw2'],
            'approved_image_byte_equality':True if a.compare else None,
            'live_capture_gpu_run_performed':False}
    write(a.output/'replay_check.json',report)
    print(report)


if __name__=='__main__':main()
