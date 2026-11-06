"""Run matched-seed trials; failures and timeouts remain in the denominator."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

CONDITIONS = {
    'nominal': [],
    'encoder_x4': ['--encoder-noise', '.002'],
    'depth_5mm': ['--depth-noise', '.005'],
    'lqr_low_effort': ['--lqr-input', '1.0'],
    'lqr_high_tracking': ['--lqr-position', '80'],
    'observer_low_q': ['--observer-q', '1e-7'],
    'observer_high_q': ['--observer-q', '1e-3'],
    'moving_red': ['--red-amplitude', '.03'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7,11,23])
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=['nominal'])
    parser.add_argument('--order', nargs='+', default=['red','blue','yellow'])
    parser.add_argument('--output', default='results/batch')
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    root = PROJECT_ROOT
    directory = (PROJECT_ROOT / args.output).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    rows = []
    for condition in args.conditions:
        for seed in args.seeds:
            run = directory/f'{condition}_seed{seed}'
            if run.exists():
                parser.error(f'{run} already exists; choose a new output folder to preserve evidence')
            run.mkdir()
            command = [sys.executable,str(root/'main_final_integration.py'),*args.order,'--headless',
                       '--seed',str(seed),'--output',str(run),'--red-amplitude','0',*CONDITIONS[condition]]
            print(f'{condition}: seed {seed}',flush=True)
            failure = ''
            with (run/'console.log').open('w',encoding='utf-8') as handle:
                try:
                    result = subprocess.run(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT,
                                            timeout=args.timeout)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    failure, code = 'process_timeout', -1
            path = run/'summary.json'
            summary = json.loads(path.read_text()) if path.exists() else {}
            row = dict(condition=condition,seed=seed,exit_code=code,
                sequence_success=bool(summary.get('sequence_success',False)) and code==0,
                failure=failure or summary.get('failure') or ('startup_or_runtime_error' if code else ''),
                completed=summary.get('completed',0))
            for key in ('joint_tracking_rmse_rad','joint_estimation_rmse_rad','torque_squared_integral',
                        'saturation_fraction','simulation_time_s'):
                row[key] = summary.get(key,'')
            rows.append(row)
            with (directory/'index.csv').open('w',newline='',encoding='utf-8') as handle:
                writer = csv.DictWriter(handle,fieldnames=list(row))
                writer.writeheader(); writer.writerows(rows)
    print(f'Results: {directory / "index.csv"}')


if __name__=='__main__':
    main()
