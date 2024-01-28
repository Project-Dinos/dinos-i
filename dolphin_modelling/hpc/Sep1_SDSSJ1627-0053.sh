#!/bin/bash
#SBATCH --job-name="Sep11627"                           
#SBATCH --output="log/Sep1_SDSSJ1627-0053.out" 
#SBATCH --error="log/Sep1_SDSSJ1627-0053.err" 
#SBATCH --partition=broadwl                                                        
#SBATCH --account=pi-jfrieman       
#SBATCH -t 36:00:00                                                              
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=16                                                            
#SBATCH --exclusive                                                            
#SBATCH --mail-user=chinyi@uchicago.edu                                          
#SBATCH --mail-type=FAIL    

module load python
conda activate astroconda

mpirun -np 16 python3 /home/chinyi/dinos/SLACS/hpc/swim.py SDSSJ1627-0053 Sep1 

