#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=tld   # the name of your job
#resources:
#SBATCH --cpus-per-task=1   # the job can use and see 4 CPUs (from max 24).
#SBATCH --partition=week    # the slurm partition the job is queued to.
#SBATCH --mem-per-cpu=30G   # the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)
#SBATCH --gres=gpu:1080ti:1      #the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=1-23:59:59   # the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=jobs/job.%J.err   # write the error output to job.*jobID*.err
#SBATCH --output=jobs/job.%J.out   # write the standard output to job.*jobID*.out
#SBATCH --mail-type=END    #write a mail if a job begins, ends, fails, gets requeued or stages out
#SBATCH --mail-user=magnus.kaut@student.uni-tuebingen.de   # your mail address

####
#c) Execute your file.
####




#nvidia-smi

#singularity run-help /common/singularityImages/TCML-CUDA12_0_TF2_12_PT1_13.simg
cd /home/stud125/TrafficLightDetection/StateDetection

singularity exec --nv /home/stud125/sdc_gym_amd64.simg python -u /home/stud125/TrafficLightDetection/StateDetection/train.py --max_keep 1200 --num_epochs 15 --log_interval 5000 --device "cuda"

#source /home/stud468/TrafficLightDetection/traffic_light_venv/bin/activate

#pip list

#python -u /home/stud468/TrafficLightDetection/StateDetection/train.py

echo DONE!
