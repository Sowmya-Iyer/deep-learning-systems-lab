"""Audit legacy BDD100K per-image JSONs and images; never changes source data.

python audit_bdd.py --root PATH/TO/bdd100k --out reports/bdd_audit --samples 50
Requires Pillow only for --samples. Native PNG masks are counted separately from
polygon annotations. Rendered QA masks are approximations, not benchmark labels.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

BAD_WEATHER = {'rainy', 'snowy', 'foggy'}
VEHICLES = {'car', 'truck', 'bus', 'train', 'motor', 'motorcycle', 'bike', 'bicycle'}
PEOPLE = {'person', 'pedestrian', 'rider'}
AREA = {'area/drivable': 0, 'area/alternative': 1, 'direct': 0, 'alternative': 1}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def csvwrite(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def seed_order(records, seed):
    """Deterministic interleaving proportional to weather x time strata."""
    strata = defaultdict(list)
    for r in records:
        strata[(r['weather'], r['timeofday'])].append(r)
    ordered = []
    for key, rows in sorted(strata.items()):
        rows = sorted(rows, key=lambda r: r['id'])
        rng = random.Random(f'{seed}:{key}')
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            ordered.append(((i + 0.5) / len(rows), key, row))
    return [r for _, _, r in sorted(ordered, key=lambda t: (t[0], t[1]))]


def objects(data):
    if 'frames' in data:
        frames = data['frames']
        if len(frames) != 1:
            raise ValueError('Expected one keyframe per JSON; sequence splitting needs explicit support')
        return frames[0].get('objects', []), frames[0].get('timestamp')
    return data.get('labels', []), data.get('timestamp')


def polygon_points(vertices):
    """Legacy L/C paths, with cubic curves sampled for visual QA only."""
    pts = []
    i = 0
    while i < len(vertices):
        x, y, code = vertices[i]
        if code in ('L', 'M'):
            pts.append((float(x), float(y)))
            i += 1
        elif code == 'C' and pts and i + 2 < len(vertices):
            if any(v[2] != 'C' for v in vertices[i:i+3]):
                raise ValueError('Malformed cubic segment')
            p0 = pts[-1]
            p1, p2, p3 = [v[:2] for v in vertices[i:i+3]]
            for j in range(1, 33):
                t = j / 32
                u = 1 - t
                pts.append(tuple(u**3*p0[k] + 3*u*u*t*p1[k] +
                                 3*u*t*t*p2[k] + t**3*p3[k] for k in (0, 1)))
            i += 3
        else:
            raise ValueError(f'Unsupported polygon command {code}')
    return pts


def previews(records, out, count, seed):
    from PIL import Image, ImageDraw
    # Round-robin condition sampling for diversity rather than prevalence.
    buckets = defaultdict(list)
    for r in records:
        if r['area_polygons']:
            buckets[(r['official_split'], r['weather'], r['timeofday'])].append(r)
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < count and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
    directory = out / 'samples'
    directory.mkdir(exist_ok=True)
    review = []
    thumbs = []
    for i, r in enumerate(selected):
        try:
            with Image.open(r['image']) as source:
                image = source.convert('RGB')
            data = json.loads(Path(r['label']).read_text(encoding='utf-8'))
            labels, _ = objects(data)
            mask = Image.new('L', image.size, 2)
            draw = ImageDraw.Draw(mask)
            for obj in labels:
                category = obj.get('category')
                if category in AREA and obj.get('poly2d'):
                    points = polygon_points(obj['poly2d'])
                    if len(points) >= 3:
                        draw.polygon(points, fill=AREA[category])
            color = Image.new('RGB', image.size)
            color.paste((230, 40, 40), mask=mask.point(lambda p: 255 if p == 0 else 0))
            color.paste((40, 90, 240), mask=mask.point(lambda p: 255 if p == 1 else 0))
            blend = Image.blend(image, color, 0.4)
            foreground = mask.point(lambda p: 255 if p in (0, 1) else 0)
            overlay = Image.composite(blend, image, foreground)
            tile = Image.new('RGB', (960, 300), 'white')
            tile.paste(image.resize((480, 270)), (0, 0))
            tile.paste(overlay.resize((480, 270)), (480, 0))
            caption = f'{i+1:02} {r["id"]} | {r["official_split"]} | {r["weather"]} | {r["timeofday"]}'
            ImageDraw.Draw(tile).text((5, 275), caption, fill='black')
            tile.save(directory / f'{i+1:02}_{r["id"]}.jpg', quality=90)
            thumbs.append(tile)
            review.append(dict(sample=i+1, id=r['id'], split=r['official_split'],
                               weather=r['weather'], timeofday=r['timeofday'],
                               width=image.width, height=image.height, status='rendered', notes=''))
        except Exception as exc:
            review.append(dict(sample=i+1, id=r['id'], split=r['official_split'],
                               weather=r['weather'], timeofday=r['timeofday'],
                               width='', height='', status=f'ERROR: {exc}', notes=''))
    for start in range(0, len(thumbs), 10):
        page = Image.new('RGB', (1920, 1500), 'white')
        for j, tile in enumerate(thumbs[start:start+10]):
            page.paste(tile, ((j % 2)*960, (j // 2)*300))
        page.save(directory / f'contact_{start//10+1:02}.jpg', quality=90)
    if review:
        csvwrite(out / 'sample_review.csv', review, list(review[0]))
    return review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('bdd_audit'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples', type=int, default=0)
    p.add_argument('--free-space', choices=['both', 'direct'], default='both')
    p.add_argument('--seconds-per-step', type=float)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--labels-zip', type=Path, help='Read JSON bytes from an identical local legacy-label ZIP for faster auditing')
    p.add_argument('--images-zip', type=Path, help='Use matching image ZIP metadata for byte sizes')
    args = p.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.samples < 0:
        p.error('batch-size and epochs must be positive; samples nonnegative')
    root = args.root.resolve()
    base = root / '100k'
    if not base.exists():
        base = root / 'images' / '100k'
    if not base.exists():
        p.error(f'No 100k/ or images/100k/ found under {root}')
    args.out.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    summary = {'root': str(root), 'seed': args.seed, 'free_space': args.free_space,
               'free_space_ids': [0, 1] if args.free_space == 'both' else [0],
               'background_id': 2, 'ignore_id': 255,
               'bad_weather': sorted(BAD_WEATHER), 'official_splits': {}}
    label_hash = hashlib.sha256()
    label_zip = zipfile.ZipFile(args.labels_zip) if args.labels_zip else None
    image_zip = zipfile.ZipFile(args.images_zip) if args.images_zip else None
    label_entries = {Path(i.filename).stem: i for i in label_zip.infolist() if i.filename.endswith('.json')} if label_zip else {}
    image_entries = {Path(i.filename).stem: i for i in image_zip.infolist() if i.filename.endswith('.jpg')} if image_zip else {}
    summary['label_read_source'] = str(args.labels_zip) if label_zip else 'local JSON files'
    summary['archive_note'] = 'When supplied, ZIP annotation bytes are audited; local paths are checked by filename, not content equivalence.'
    for split in ['train', 'val', 'test']:
        images = {f.stem: f for f in (base / split).glob('*.jpg')}
        labels = {f.stem: f for f in (base / split).glob('*.json')}
        stats = dict(images=len(images), jsons=len(labels), image_bytes=0,
                     json_bytes=0, images_without_json=len(images.keys()-labels.keys()),
                     jsons_without_images=len(labels.keys()-images.keys()))
        for index, (stem, image) in enumerate(sorted(images.items())):
            if index % 5000 == 0:
                print(f'Reading {split}: {index}/{len(images)}', flush=True)
            stats['image_bytes'] += image_entries[stem].file_size if image_zip else image.stat().st_size
            if stem not in labels:
                continue
            path = labels[stem]
            try:
                raw = label_zip.read(label_entries[stem]) if label_zip else path.read_bytes()
                stats['json_bytes'] += len(raw)
                label_hash.update(f'{split}/{stem}\0'.encode())
                label_hash.update(hashlib.sha256(raw).digest())
                data = json.loads(raw)
                labels_here, timestamp = objects(data)
                attrs = data.get('attributes', {})
                areas = [o for o in labels_here if o.get('category') in AREA and o.get('poly2d')]
                boxes = [o for o in labels_here if o.get('box2d')]
                valid_boxes, invalid_boxes = [], 0
                for obj in boxes:
                    box = obj['box2d']
                    vals = [box.get(k) for k in ['x1', 'y1', 'x2', 'y2']]
                    valid = (all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
                             and vals[2] > vals[0] and vals[3] > vals[1])
                    if valid:
                        valid_boxes.append(obj)
                    else:
                        invalid_boxes += 1
                group = data.get('videoName') or data.get('video_name') or stem
                records.append(dict(id=stem, sequence=str(group), official_split=split,
                    image=str(image), label=str(path), weather=attrs.get('weather', 'missing'),
                    timeofday=attrs.get('timeofday', 'missing'), scene=attrs.get('scene', 'missing'),
                    timestamp=timestamp, area_polygons=len(areas),
                    direct_polygons=sum(o['category'] in ['area/drivable', 'direct'] for o in areas),
                    alternative_polygons=sum(o['category'] in ['area/alternative', 'alternative'] for o in areas),
                    boxes=len(valid_boxes), invalid_boxes=invalid_boxes,
                    vehicle_boxes=sum(o.get('category') in VEHICLES for o in valid_boxes),
                    person_boxes=sum(o.get('category') in PEOPLE for o in valid_boxes)))
            except Exception as exc:
                errors.append(dict(path=str(path), error=str(exc)))
        split_records = [r for r in records if r['official_split'] == split]
        stats.update(images_with_polygons=sum(r['area_polygons'] > 0 for r in split_records),
                     images_with_boxes=sum(r['boxes'] > 0 for r in split_records),
                     invalid_boxes=sum(r['invalid_boxes'] for r in split_records))
        summary['official_splits'][split] = stats
        print(f'{split}: {stats}', flush=True)
    if not records:
        p.error('No supported per-image JSON/image pairs found. This script expects the legacy layout.')
    if errors:
        dump(args.out / 'parse_errors.json', errors)
        raise ValueError(f'{len(errors)} malformed annotations; fix before freezing splits')
    # An image stem represents one source video in the official keyframe package.
    # Explicit video metadata takes precedence; repeated video IDs fail closed.
    group_members = defaultdict(list)
    for r in records:
        group_members[r['sequence']].append(r)
    repeated = {g: [r['image'] for r in rows] for g, rows in group_members.items() if len(rows) > 1}
    if repeated:
        dump(args.out / 'repeated_sequences.json', repeated)
        raise ValueError('Repeated source-video IDs found; use a group-aware splitter before proceeding')
    train = [r for r in records if r['official_split'] == 'train']
    ordered_train = seed_order(train, args.seed)
    official_val = seed_order([r for r in records if r['official_split'] == 'val'], args.seed+1)
    validation = official_val[::2]
    test = official_val[1::2]
    train_ids = [r['id'] for r in ordered_train]
    n10, n25 = round(len(train_ids)*0.10), round(len(train_ids)*0.25)
    protocol = dict(
        status='frozen image-ID protocol; raster-mask readiness must be verified separately',
        seed=args.seed, annotation_sha256=label_hash.hexdigest(),
        sequence_rule='explicit videoName/video_name, otherwise full keyframe basename (not prefix)',
        protocol='official train; official val divided into custom validation/test; official test excluded',
        free_space=args.free_space, bad_weather=sorted(BAD_WEATHER),
        train=train_ids, validation=[r['id'] for r in validation], test=[r['id'] for r in test],
        label_subsets={'10': train_ids[:n10], '25': train_ids[:n25], '100': train_ids},
        ssl_pretraining=train_ids,
        object_metric=dict(type='box-bottom proxy, not true object-pixel overlap', bottom_fraction=0.25,
                           categories=sorted(VEHICLES | PEOPLE),
                           definition='predicted-free pixels / union of bottom-quarter box pixels; exclude ignored/crowd objects and report vehicle/person separately'))
    a, b, c = [set(protocol[k]) for k in ['train', 'validation', 'test']]
    assert not (a & b or a & c or b & c)
    assert set(protocol['label_subsets']['10']) <= set(protocol['label_subsets']['25']) <= a
    assert not set(protocol['ssl_pretraining']) & (b | c)
    splitfile = args.out / 'bdd_split.json'
    if splitfile.exists():
        old = json.loads(splitfile.read_text(encoding='utf-8'))
        if old != protocol:
            raise ValueError('Existing frozen split differs; use a new output directory instead of overwriting')
    dump(splitfile, protocol)
    memberships = {'train': train, 'validation': validation, 'test': test,
                   'train_10': ordered_train[:n10], 'train_25': ordered_train[:n25]}
    counts = []
    for split, rows in memberships.items():
        for field in ['weather', 'timeofday', 'joint']:
            counter = Counter((r['weather']+' | '+r['timeofday']) if field == 'joint' else r[field] for r in rows)
            for condition, n in sorted(counter.items()):
                counts.append(dict(split=split, tag=field, condition=condition, images=n))
        for condition, selected in [('bad_weather', [r for r in rows if r['weather'] in BAD_WEATHER]),
                                    ('night_bad_weather', [r for r in rows if r['weather'] in BAD_WEATHER and r['timeofday']=='night'])]:
            counts.append(dict(split=split, tag='derived', condition=condition, images=len(selected)))
    csvwrite(args.out/'condition_counts.csv', counts, ['split','tag','condition','images'])
    csvwrite(args.out/'image_inventory.csv', records, list(records[0]))
    object_coverage = {}
    for split, rows in memberships.items():
        object_coverage[split] = dict(images=len(rows),
            with_vehicle_boxes=sum(r['vehicle_boxes'] > 0 for r in rows),
            with_person_boxes=sum(r['person_boxes'] > 0 for r in rows),
            with_either=sum(r['vehicle_boxes']+r['person_boxes'] > 0 for r in rows),
            with_both=sum(r['vehicle_boxes'] > 0 and r['person_boxes'] > 0 for r in rows),
            with_area_polygons=sum(r['area_polygons'] > 0 for r in rows))
    summary['object_coverage'] = object_coverage
    summary['annotation_sha256'] = label_hash.hexdigest()
    summary['raster_drivable_masks'] = {}
    for split in ['train','val','test']:
        folder = root/'labels'/'drivable'/'masks'/split
        masks = {f.stem for f in folder.glob('*.png')}
        ids = {r['id'] for r in records if r['official_split'] == split}
        summary['raster_drivable_masks'][split] = dict(files=len(masks), matched=len(ids & masks))
    summary['instance_bitmask_files'] = sum(1 for _ in (root/'labels'/'ins_seg'/'bitmasks').rglob('*.png'))
    summary['checks'] = dict(sequence_overlap=0, nested_subsets=True, ssl_holdout_overlap=0,
                             visual_duplicates='not checked', raster_masks='not decoded')
    summary['condition_sufficiency_rule'] = 'Flag groups below 100 images; heuristic, not a statistical power calculation'
    summary['low_count_groups'] = [r for r in counts if r['images'] < 100]
    summary['training_resolution'] = [640, 360]
    steps = 3*3*args.epochs*sum(math.ceil(n/args.batch_size) for n in [n10,n25,len(train)])
    summary['compute'] = dict(runs=27, methods=3, seeds=3, epochs=args.epochs,
        batch_size=args.batch_size, optimizer_steps=steps,
        assumed_seconds_per_step=args.seconds_per_step,
        training_hours=steps*args.seconds_per_step/3600 if args.seconds_per_step else None,
        excludes='validation, preprocessing, SSL pretraining, I/O overhead; benchmark actual model first',
        note='Four methods in project report would require 36 runs, not 27')
    if args.samples:
        review = previews(records, args.out, args.samples, args.seed)
        summary['qa_samples'] = len(review)
        summary['qa_render_errors'] = sum(r['status'] != 'rendered' for r in review)
        summary['sample_resolutions'] = dict(Counter(f'{r["width"]}x{r["height"]}' for r in review))
    dump(args.out/'audit_summary.json', summary)
    print(f'Wrote audit to {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
