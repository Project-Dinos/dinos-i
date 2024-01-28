#!/usr/bin/env bash


function usage {
    echo -e "\nUsage:\n $0 <config_string>"
}

if [ $# == 0 ]; then
    echo -e "\n Please provide a config_string"
    usage
    exit
fi

run_id="$1"
lens_name="$2"
short_name="${lens_name:5:4}" 

base_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cat << EOF > ./${run_id}_${lens_name}.sh
#!/bin/bash
#SBATCH --job-name="${run_id}${short_name}"                           
#SBATCH --output="log/${run_id}_${lens_name}.out" 
#SBATCH --error="log/${run_id}_${lens_name}.err" 
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

mpirun -np 16 python3 /home/chinyi/backup/Oct_2022/SLACS/hpc/swim.py $lens_name $run_id 

EOF

chmod u+x ${run_id}_${lens_name}.sh

if [[ -x ${run_id}_${lens_name}.sh ]]; then
    echo "sbatch ${run_id}_${lens_name}.sh"
    sbatch ${run_id}_${lens_name}.sh
fi