"""Portable, explicit run preparation; never render or start training."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(manifest):
    content = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def path_maps(values):
    result = []
    for value in values:
        old, separator, new = value.partition('=')
        if (not separator or old == '/' or new == '/' or
                not Path(old).is_absolute() or not Path(new).is_absolute()):
            raise ValueError('Use --path-map /old/root=/new/root with absolute paths')
        result.append((old.rstrip('/'), new.rstrip('/')))
    return sorted(result, key=lambda pair: len(pair[0]), reverse=True)


def remap(value, mappings):
    if isinstance(value, dict):
        return {key: remap(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item, mappings) for item in value]
    if isinstance(value, str):
        for old, new in mappings:
            if value == old or value.startswith(old + '/'):
                return new + value[len(old):]
    return value


def relocated_manifest(manifest, mappings):
    if manifest.get('manifest_sha256') != manifest_digest(manifest):
        raise ValueError('Original manifest hash mismatch')
    result = remap(copy.deepcopy(manifest), mappings)
    result['manifest_sha256'] = manifest_digest(result)
    # Every non-path label, split, frame identity and ordering is preserved by remap.
    return result


def verify_files(manifest):
    """Verify every referenced RGB, mask and label, including duplicate references."""
    verified = {}
    for record in manifest['records']:
        required = [(record['label_path'], record['label_sha256'])]
        for paths, hashes in (('image_paths', 'image_sha256'), ('mask_paths', 'mask_sha256')):
            if set(record[paths]) != {'left', 'right'} or set(record[hashes]) != {'left', 'right'}:
                raise ValueError('Both stereo views and their hashes are required')
            required.extend((record[paths][eye], record[hashes][eye]) for eye in ('left', 'right'))
        for filename, expected in required:
            if filename in verified:
                if verified[filename] != expected:
                    raise ValueError('Conflicting hashes for ' + filename)
            elif sha256(filename) != expected:
                raise ValueError('Content hash mismatch: ' + filename)
            else:
                verified[filename] = expected
    if not verified:
        raise ValueError('Empty dataset')
    return len(verified)


def empty_destination(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Choose an empty output directory: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
