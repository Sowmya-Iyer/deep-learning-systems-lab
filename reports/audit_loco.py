"""Audit LOCO candidate crops without downloading images.

Example: python audit_loco.py --repo loco --min-side 32
Counts refer to annotation boxes, not visually verified extracted crops.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('loco'))
    parser.add_argument('--out', type=Path, default=Path('loco_audit'))
    parser.add_argument('--min-side', type=float, default=32)
    parser.add_argument('--split-file', type=Path,
                        help='JSON mapping subset numbers to train/validation/test')
    args = parser.parse_args()
    if not math.isfinite(args.min_side) or args.min_side <= 0:
        parser.error('--min-side must be a finite positive number')

    # Example custom split, NOT an official train/validation/test benchmark.
    assignment = {1: 'validation', 2: 'train', 3: 'train', 4: 'test', 5: 'train'}
    if args.split_file:
        assignment = {int(k): v for k, v in
                      json.loads(args.split_file.read_text(encoding='utf-8')).items()}
    splits = ['train', 'validation', 'test']
    if set(assignment) != set(range(1, 6)) or set(assignment.values()) != set(splits):
        parser.error('Assign each subset 1–5 once and use all three split names')

    before, after, image_counts = Counter(), Counter(), Counter()
    classes, seen, manifest = {}, {}, []
    invalid = 0
    for subset in range(1, 6):
        files = sorted((args.repo / 'rgb').glob(f'loco-sub{subset}-v1-*.json'))
        if len(files) != 1:
            parser.error(f'Expected one subset-{subset} JSON in {args.repo / "rgb"}; found {len(files)}')
        data = json.loads(files[0].read_text(encoding='utf-8'))
        for category in data['categories']:
            cid, name = category['id'], category['name']
            if cid in classes and classes[cid] != name:
                raise ValueError(f'Inconsistent category ID: {cid}')
            classes[cid] = name
        images = {image['id']: image for image in data['images']}
        if len(images) != len(data['images']):
            raise ValueError(f'Duplicate image IDs in {files[0]}')
        for image in images.values():
            source = image['path']
            if source in seen:
                raise ValueError(f'Repeated source path in subsets {seen[source]} and {subset}: {source}')
            seen[source] = subset
            image_counts[subset] += 1

        for annotation in data['annotations']:
            image = images[annotation['image_id']]
            name = classes[annotation['category_id']]
            before[subset, name] += 1
            x, y, width, height = map(float, annotation['bbox'])
            iw, ih = float(image['width']), float(image['height'])
            valid = (all(math.isfinite(v) for v in (x, y, width, height, iw, ih))
                     and width > 0 and height > 0 and iw > 0 and ih > 0)
            x1 = y1 = x2 = y2 = 0.0
            if valid:
                x1, y1 = max(0, min(iw, x)), max(0, min(ih, y))
                x2, y2 = max(0, min(iw, x + width)), max(0, min(ih, y + height))
                valid = x2 > x1 and y2 > y1
            keep = valid and min(x2 - x1, y2 - y1) >= args.min_side
            invalid += int(not valid)
            if keep:
                after[subset, name] += 1
            manifest.append(dict(
                subset=subset, split=assignment[subset], image_id=image['id'],
                annotation_id=annotation['id'], source_path=image['path'],
                class_name=name, x1=x1, y1=y1, x2=x2, y2=y2,
                keep=int(keep), valid_bbox=int(valid),
                iscrowd=int(bool(annotation.get('iscrowd', 0))),
            ))

    names = sorted(set(classes.values()))
    group_rows = [dict(subset=s, split=assignment[s], images=image_counts[s],
                       class_name=n, before=before[s, n], after=after[s, n])
                  for s in range(1, 6) for n in names]
    split_rows = []
    for split in splits:
        groups = [s for s in assignment if assignment[s] == split]
        for name in names:
            split_rows.append(dict(
                split=split, class_name=name,
                before=sum(before[s, name] for s in groups),
                after=sum(after[s, name] for s in groups)))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'counts_by_subset.csv', group_rows, list(group_rows[0]))
    write_csv(args.out / 'counts_by_split.csv', split_rows, list(split_rows[0]))
    fields = ['subset', 'split', 'image_id', 'annotation_id', 'source_path',
              'class_name', 'x1', 'y1', 'x2', 'y2', 'keep', 'valid_bbox', 'iscrowd']
    write_csv(args.out / 'crop_manifest.csv', manifest, fields)
    metadata = dict(min_side=args.min_side, assignment=assignment,
                    source_images=len(seen), boxes_before=sum(before.values()),
                    boxes_after=sum(after.values()), invalid_boxes=invalid)
    (args.out / 'audit_settings.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print('\nRetained candidate crops:')
    for row in split_rows:
        print(f'{row["split"]:12} {row["class_name"]:22} {row["after"]:8}')
    missing = [(r['split'], r['class_name']) for r in split_rows if r['after'] == 0]
    print('\nMissing classes:', missing or 'None')
    print('Source paths are disjoint across subsets; visual duplicates were not checked.')
    print('Counts include crowd annotations if present; inspect iscrowd before extraction.')
    print('No images were opened or crops extracted. Sequence identities were not inferred.')
    print(f'Reports: {args.out.resolve()}')


if __name__ == '__main__':
    main()
