import shutil
import os
import json

from src.helpers import environmentvariables
from src.helpers import out_dir_config

#VARS
min_iter_sac = 200

#Get all directories
environmentvariables.initialize()
out_dir = out_dir_config.get_out_dir(__file__)
experiments_dir = os.getenv("OUT_DIR")

if not os.path.exists(os.path.join(out_dir, "hdf5_files")):
    os.makedirs(os.path.join(out_dir, "hdf5_files"))

experiments_folders = []
#Identify all folders which contain a relevant sac-experiment
for experiment in os.listdir(experiments_dir):
    exp_dir = os.path.join(experiments_dir, experiment)
    if os.path.isdir(exp_dir):
        exp_dir_list = os.listdir(exp_dir)
        if 'hdf_files' in exp_dir_list and ('itr_' + str(min_iter_sac) + '.pkl') in exp_dir_list:
            experiments_folders.append(exp_dir)


#Iterate over experiments and generate configfile and copy files
env_mapping = {}
for idx, experiment in enumerate(experiments_folders):
    #first read in variant to get env
    if 'variant.json' not in os.listdir(experiment):
        continue
    with open(os.path.join(experiment, 'variant.json')) as variant:
        variant_content = json.load(variant)
        env_mapping[idx] = variant_content['env_name']
        shutil.copyfile(os.path.join(experiment, 'hdf_files', 'paths.hdf5'), os.path.join(out_dir, str(idx)+'.hdf5'))

with open(os.path.join(out_dir, 'env_mapping.json'), 'w') as env_mapping_file:
    json.dump(env_mapping, env_mapping_file)

print("Done!!")