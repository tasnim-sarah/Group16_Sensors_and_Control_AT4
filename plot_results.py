"""Produce report figures from recorded evidence, never synthetic result rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    path=args.directory
    if (path/'object_tracking.csv').exists():
        data=np.genfromtxt(path/'object_tracking.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
        data=np.atleast_1d(data)
        red=data[data['label']=='Red Cube']
        if len(red):
            summary=json.loads((path/'summary.json').read_text()) if (path/'summary.json').exists() else {}
            contacts=[e for e in summary.get('events',[]) if e.get('cube')=='Red Cube' and e['event']=='first_finger_contact_evaluation']
            if contacts:
                red=red[red['time_s']<=contacts[0]['time_s']+.8]
            fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
            for i,axis in enumerate(('x','y')):
                t=red['time_s']
                axes[0,i].plot(t,red['truth_position_'+axis],label='Evaluation truth',color='black',linewidth=1)
                axes[0,i].scatter(t,red['raw_'+axis],label='Raw RGB-D',s=5,alpha=.45)
                axes[0,i].plot(t,red['filtered_'+axis],label='Kalman estimate',linewidth=1)
                sigma=red['sigma_'+axis]
                axes[0,i].fill_between(t,red['filtered_'+axis]-2*sigma,red['filtered_'+axis]+2*sigma,alpha=.15,label='KF ±2 sigma')
                axes[0,i].set_ylabel(axis+' position (m)')
                axes[1,i].plot(t,red['truth_velocity_'+axis]*1000,label='Evaluation truth',color='black')
                axes[1,i].plot(t,red['velocity_'+axis]*1000,label='Kalman velocity')
                axes[1,i].set_ylabel(axis+' velocity (mm/s)');axes[1,i].set_xlabel('Simulation time (s)')
                for ax in axes[:,i]:
                    if contacts:ax.axvline(contacts[0]['time_s'],color='red',linestyle='--',label='First finger contact')
                    ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=7,ncol=2);axes[1,0].legend(fontsize=7)
            fig.suptitle('Moving red cube: measured tracking through approach and closure')
            fig.tight_layout();fig.savefig(path/'live_tracking.png',dpi=160);plt.close(fig)
    if (path/'trajectory.csv').exists():
        data=np.genfromtxt(path/'trajectory.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
        data=np.atleast_1d(data)
        t=data['time_s']
        fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
        estimate=data['estimate_q_1'];sigma=data['sigma_q_1']
        axes[0].plot(t,data['reference_1'],label='Reference',alpha=.7)
        axes[0].plot(t,data['truth_q_1'],label='Evaluation truth')
        axes[0].plot(t,estimate,label='KF estimate',linewidth=.8)
        axes[0].fill_between(t,estimate-2*sigma,estimate+2*sigma,alpha=.25,label='KF Â±2Ïƒ')
        axes[0].set_ylabel('Joint 1 (rad)');axes[0].legend(ncol=2)
        for i in range(1,7):
            axes[1].plot(t,data[f'truth_q_{i}']-data[f'reference_{i}'],label=f'Joint {i}')
            axes[2].plot(t,data[f'torque_{i}'],label=f'Joint {i}')
        axes[1].set_ylabel('Tracking error (rad)');axes[1].legend(ncol=3)
        axes[2].set_ylabel('Actuator torque (N m)');axes[2].set_xlabel('Simulation time (s)')
        for ax in axes:ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(path/'tracking.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(10,7),sharex=True,sharey=True)
        stride=max(1,len(t)//2500)
        chosen=np.arange(len(t))[t>=.1][::stride]
        tt=t[chosen]
        for i,ax in enumerate(axes.flat,1):
            sigma=data[f'sigma_q_{i}'][chosen]*1000
            error=(data[f'estimate_q_{i}']-data[f'truth_q_{i}'])[chosen]*1000
            ax.fill_between(tt,-2*sigma,2*sigma,alpha=.3,label='KF ±2σ')
            ax.plot(tt,error,linewidth=.4,label='Estimate minus truth')
            ax.set_title(f'Joint {i}');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Estimation errors and KF uncertainty (first 0.1 s of initialisation omitted)')
        fig.supxlabel('Simulation time (s)');fig.supylabel('Joint estimation error (mrad)')
        fig.tight_layout();fig.savefig(path/'uncertainty.png',dpi=160);plt.close(fig)
        from robot_scene import build_scene
        model,_=build_scene(7)
        torques=np.column_stack([data[f'torque_{i}'] for i in range(1,7)])
        limits=model.actuator_forcerange[:6]
        active=(torques<=limits[:,0]+1e-6)|(torques>=limits[:,1]-1e-6)
        metrics={'force_limit_fraction_any_joint':float(np.mean(np.any(active,axis=1))),
                 'force_limit_fraction_per_joint':np.mean(active,axis=0).tolist(),
                 'note':'Force-limit occupancy, separate from actuator-target clipping in summary.json'}
        (path/'force_limits.json').write_text(json.dumps(metrics,indent=2))
    if (path/'measurements.csv').exists():
        data=np.genfromtxt(path/'measurements.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
        fig,ax=plt.subplots(figsize=(7,5))
        scatter=ax.scatter(data['known_x'],data['known_y'],c=data['error_3d_m']*1000,
                           vmin=0,s=120,edgecolors='k')
        fig.colorbar(scatter,ax=ax,label='3D localisation error (mm)')
        ax.set(xlabel='Known x (m)',ylabel='Known y (m)',title='Static simulated camera validation')
        ax.set_aspect('equal');fig.tight_layout();fig.savefig(path/'localisation.png',dpi=160);plt.close(fig)
    if (path/'summary.json').exists():
        summary=json.loads((path/'summary.json').read_text())
        if summary.get('bench')=='local_joint_step':
            rows=summary['results']
            fig,axes=plt.subplots(2,2,figsize=(11,8))
            noises=sorted({r['encoder_noise_std_rad'] for r in rows})
            panels=[(['baseline','tracking_weight_80','input_weight_1'],['Baseline','Position 80','Input 1'],
                     'tracking_rmse_rad',1000,'Tracking RMSE (mrad)'),
                    (['baseline','tracking_weight_80','input_weight_1'],['Baseline','Position 80','Input 1'],
                     'torque_squared_integral',1,'Torque squared integral (N^2 m^2 s)'),
                    (['baseline','observer_q_1e-7','observer_q_1e-3'],['Q = 1e-5','Q = 1e-7','Q = 1e-3'],
                     'estimation_rmse_rad',1000,'Angle estimation RMSE (mrad)'),
                    (['baseline','observer_q_1e-7','observer_q_1e-3'],['Q = 1e-5','Q = 1e-7','Q = 1e-3'],
                     'velocity_estimation_rmse_rad_s',1000,'Velocity estimation RMSE (mrad/s)')]
            for ax,(settings,labels,metric,scale,ylabel) in zip(axes.flat,panels):
                for index,noise in enumerate(noises):
                    values=[next(r[metric]*scale for r in rows if r['condition']==setting and r['encoder_noise_std_rad']==noise) for setting in settings]
                    bars=ax.bar(np.arange(3)+(index-.5)*.35,values,width=.35,label=f'Encoder noise std = {noise*1000:g} mrad')
                    ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
                ax.set_xticks(np.arange(3),labels);ax.set_ylabel(ylabel);ax.set_ylim(0,ax.get_ylim()[1]*1.16)
                ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            axes[0,0].legend(fontsize=8)
            fig.suptitle('Local joint-step study: three matched seeds per setting/noise level')
            fig.tight_layout();fig.savefig(path/'tradeoffs.png',dpi=160);plt.close(fig)
    print(f'Figures written to {path}')


if __name__=='__main__':
    main()
