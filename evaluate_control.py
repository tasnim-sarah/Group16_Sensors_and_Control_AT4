"""Matched-seed local joint-step experiments on the actual MuJoCo actuator plant.

This isolates control/estimation near HOME; it does not establish sorting robustness.
"""
import argparse
import csv
import json
from pathlib import Path
import mujoco
import numpy as np
from robot_scene import HOME,build_scene
from controlfile import PositionActuatorStateSpaceModel,PositionActuatorLQRController,JointKalmanObserver

SETTINGS={
    'baseline':(20.,.1,1e-5),
    'tracking_weight_80':(80.,.1,1e-5),
    'input_weight_1':(20.,1.,1e-5),
    'observer_q_1e-7':(20.,.1,1e-7),
    'observer_q_1e-3':(20.,.1,1e-3),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/local_control_study'))
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,11,23])
    parser.add_argument('--noise',nargs='+',type=float,default=[.0005,.002])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,(position_weight,input_weight,process_noise) in SETTINGS.items():
        for noise in args.noise:
            for seed in args.seeds:
                model,data=build_scene(seed)
                dynamics=PositionActuatorStateSpaceModel(model,data)
                controller=PositionActuatorLQRController(dynamics,position_weight=position_weight,input_weight=input_weight)
                observer=JointKalmanObserver(dynamics,position_noise=noise**2,process_noise=process_noise)
                rng=np.random.default_rng(seed+1000)
                q,dq=observer.reset(HOME+rng.normal(0,noise,6))
                command=HOME.copy()
                track,estimate,velocity,effort,nis,coverage,saturations=[],[],[],[],[],[],[]
                for step in range(1250):
                    if step:
                        q,dq=observer.update(data.qpos[:6]+rng.normal(0,noise,6),command)
                    reference=HOME.copy()
                    if data.time>=.25:
                        reference+=np.array([.04,-.02,.02,0,0,0])
                    raw=controller.raw_command(reference,q,dq)
                    command=controller.command(reference,q,dq)
                    data.ctrl[:6]=command
                    track.append(data.qpos[:6]-reference)
                    estimate.append(q-data.qpos[:6])
                    velocity.append(dq-data.qvel[:6])
                    coverage.extend(abs(q-data.qpos[:6])<=2*observer.position_standard_deviation())
                    effort.append(data.actuator_force[:6].copy())
                    nis.append(getattr(observer,'last_nis',0.))
                    saturations.append(np.any(abs(command-raw)>1e-10))
                    mujoco.mj_step(model,data)
                row=dict(condition=label,seed=seed,encoder_noise_std_rad=noise,
                    lqr_position_weight=position_weight,lqr_input_weight=input_weight,observer_q=process_noise,
                    tracking_rmse_rad=float(np.sqrt(np.mean(np.asarray(track)**2))),
                    estimation_rmse_rad=float(np.sqrt(np.mean(np.asarray(estimate)**2))),
                    velocity_estimation_rmse_rad_s=float(np.sqrt(np.mean(np.asarray(velocity)**2))),
                    steady_tracking_rmse_rad=float(np.sqrt(np.mean(np.asarray(track)[-250:]**2))),
                    torque_squared_integral=float(np.sum(np.asarray(effort)**2)*model.opt.timestep),
                    saturation_fraction=float(np.mean(saturations)),
                    marginal_2sigma_coverage=float(np.mean(coverage)),nis_mean=float(np.mean(nis)))
                rows.append(row)
                print(label,noise,seed,'RMSE',round(row['tracking_rmse_rad'],6),flush=True)
    with (args.output/'trials.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for label in SETTINGS:
        for noise in args.noise:
            group=[r for r in rows if r['condition']==label and r['encoder_noise_std_rad']==noise]
            result={'condition':label,'encoder_noise_std_rad':noise,'trials':len(group)}
            for key in ('tracking_rmse_rad','estimation_rmse_rad','velocity_estimation_rmse_rad_s',
                        'steady_tracking_rmse_rad','torque_squared_integral','saturation_fraction',
                        'marginal_2sigma_coverage','nis_mean'):
                result[key]=float(np.mean([r[key] for r in group]))
            summary.append(result)
    (args.output/'summary.json').write_text(json.dumps({'bench':'local_joint_step','results':summary},indent=2))


if __name__=='__main__':
    main()
