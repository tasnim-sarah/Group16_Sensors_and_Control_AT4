"""Static simulation calibration bench: known grid locations, noisy depth/encoders.

This deliberately sets scene poses; it is a sensor validation, not a sorting run.
Localisation uses only noisy estimated-joint FK, images and known cube geometry.
"""
import argparse
import csv
import json
from pathlib import Path
import mujoco
import numpy as np
from robot_pipeline import SortingRobot, parse_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/camera_validation')
    parser.add_argument('--depth-noise',type=float,default=0)
    parser.add_argument('--samples',type=int,default=5)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    robot = SortingRobot(parse_args(['--headless','--depth-noise',str(args.depth_noise)]))
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    rows,errors = [],[]
    try:
        q,ok = robot.ik(np.array([.32,0,.25]),robot.home_rotation,robot.q_est)
        if not ok:
            raise RuntimeError('camera bench pose is unreachable')
        # BENCH SETUP ONLY: known robot pose and known object coordinates.
        robot.data.qpos[:6]=q
        for colour in ('red','blue'):
            other=int(robot.model.jnt_qposadr[robot.model.joint(colour+'_cube_joint').id])
            robot.data.qpos[other:other+3]=[-.5,-.5,.02]
        adr=int(robot.model.jnt_qposadr[robot.model.joint('yellow_cube_joint').id])
        for x in (.28,.32,.36):
            for y in (-.12,0,.12):
                robot.data.qpos[adr:adr+3]=[x,y,.020]
                for sample in range(args.samples):
                    robot.data.time += robot.camera_steps*robot.dt
                    mujoco.mj_forward(robot.model,robot.data)
                    robot.q_est=q+robot.rng.normal(0,robot.args.encoder_noise,6)
                    robot.capture()
                    measured=robot.raw.get('Yellow Cube')
                    error=np.full(3,np.nan) if measured is None else measured-np.array([x,y,.020])
                    rows.append([x,y,.020,sample,measured is not None,*error,float(np.linalg.norm(error))])
                    if measured is not None:
                        errors.append(error)
        with (output/'measurements.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['known_x','known_y','known_z','sample','detected','error_x','error_y','error_z','error_3d_m']);writer.writerows(rows)
        errors=np.asarray(errors)
        result={'bench':'static_simulation_known_grid','attempted':len(rows),'detected':len(errors),
                'source_sha256':robot.source_hashes,
                'detection_fraction':len(errors)/len(rows),'depth_noise_std_m':args.depth_noise,
                'encoder_noise_std_rad':robot.args.encoder_noise,
                'rmse_3d_m':float(np.sqrt(np.mean(np.sum(errors**2,axis=1)))) if len(errors) else None,
                'bias_xyz_m':np.mean(errors,axis=0).tolist() if len(errors) else None,
                'max_error_m':float(np.max(np.linalg.norm(errors,axis=1))) if len(errors) else None}
        (output/'summary.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        robot.close()


if __name__=='__main__':
    main()
