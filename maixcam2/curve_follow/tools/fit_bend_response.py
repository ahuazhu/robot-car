"""Rebuild the bounded paired-response mesh from archived accepted measurements."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'docs/data/rolling-turn-response-20261112'
OUTPUT = ROOT/'maixcam2/curve_follow/robot/navigation/bend_response_data.json'


def build():
    grouped = defaultdict(list)
    sources = []
    for name in ('first-audit.json', 'high-audit.json', 'retest-audit.json'):
        path = DATA/name
        sources.append(dict(file=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for row in json.loads(path.read_text())['trials']:
            if row['full_trial_eligible_for_static_fit']:
                grouped[tuple(row['actual_pwm'])].append((name, row))
    straight_path = DATA/'straight-anchors.json'
    straight = json.loads(straight_path.read_text())['anchors']
    sources.append(dict(file=straight_path.name, sha256=hashlib.sha256(straight_path.read_bytes()).hexdigest()))
    low_path = DATA/'low-anchors.json'
    low = json.loads(low_path.read_text())['trials']
    sources.append(dict(file=low_path.name, sha256=hashlib.sha256(low_path.read_bytes()).hexdigest()))
    result = dict(schema=1, sources=sources,
        method='Equal weight per accepted rolling trial; interpolate body v,w inside explicit triangles, then divide w/v. No monotonic curvature fit or extrapolation.',
        limitations=['Inside70 row is directly sampled; other combinations are interpolation estimates.',
                     'Straight anchors use older terminal steady windows, with measured nonzero yaw; battery conditions across dates were not confirmed.',
                     'Lower-domain120/60 and60/120 anchors are old cold-start late windows;140/50 is rolling. They remain distinct from the new inside70 curve.',
                     '255/70 skipped by user and excluded. Optical body motion is not wheel RPM.'])
    for name, side in (('right', 0), ('left', 1)):
        nodes = []
        for point in straight:
            duty = point['pwm'][0]
            nodes.append(dict(value=[duty, duty, point['forward_m_s'], point['yaw_rad_s']],
                              source=['straight-anchors.json', duty]))
        diagonal = list(range(len(nodes)))
        row_ids = []
        for pair, trials in sorted(grouped.items(), key=lambda item:max(item[0])):
            if pair[side] <= pair[1-side]:
                continue
            assert pair[1-side] == 70 and pair != (255,70)
            row_ids.append(len(nodes))
            values = [row['windows']['0.3'] for _,row in trials]
            nodes.append(dict(value=[pair[side],70,
                sum(x['forward_m_s'] for x in values)/len(values),
                sum(x['yaw_rad_s'] for x in values)/len(values)],
                source=[[filename,row['id']] for filename,row in trials]))
        # Two triangle fans fill the polygon between the measured diagonal and
        # inner70 row. Every edge uses shared nodes, so the surface is continuous.
        triangles = [[row_ids[0], a, b] for a,b in zip(diagonal, diagonal[1:])]
        triangles += [[diagonal[-1], a, b] for a,b in zip(row_ids, row_ids[1:])]
        low_ids = []
        for record in low:
            pair = record['actual_pwm']
            if pair[side] <= pair[1-side]:
                continue
            w = record['windows']['0.3']
            low_ids.append(len(nodes))
            nodes.append(dict(value=[pair[side],pair[1-side],w['forward_m_s'],w['yaw_rad_s']],
                source=['low-anchors.json',record['id']],warmup_s=record['warmup_s']))
        # Extend the lower boundary using actual lower-duty observations. This
        # avoids a discontinuous on/off boundary at inner70 without clamping
        # the inside wheel or inventing unmeasured low-duty responses.
        triangles.append([low_ids[0],diagonal[0],row_ids[0]])
        if len(low_ids) == 2:
            triangles.append([low_ids[0],low_ids[1],row_ids[0]])
        triangles += [[low_ids[-1],a,b] for a,b in zip(row_ids,row_ids[1:])]
        result[name] = dict(nodes=nodes, triangles=triangles)
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    result=build()
    print('Saved paired response:', {name:len(result[name]['nodes']) for name in ('left','right')})
