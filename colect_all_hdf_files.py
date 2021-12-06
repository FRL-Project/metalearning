import shutil
import os
import json

from src.helpers import environmentvariables
from src.helpers import out_dir_config

#VARS
min_iter_sac = 50
create_ml10_config = True

#Get all directories
environmentvariables.initialize()
out_dir = out_dir_config.get_out_dir(__file__)
experiments_dir = os.getenv("OUT_DIR")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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
        env_mapping[variant_content['env_name']] = idx
        shutil.copyfile(os.path.join(experiment, 'hdf_files', 'paths.hdf5'), os.path.join(out_dir, str(idx)+'.hdf5'))
        if os.path.isfile(os.path.join(experiment, 'progress.csv')):
            shutil.copyfile(os.path.join(experiment, 'progress.csv'), os.path.join(out_dir, str(idx)+'.csv'))

with open(os.path.join(out_dir, 'env_mapping_sac_training.json'), 'w') as env_mapping_file:
    json.dump(env_mapping, env_mapping_file)

#generate task_config for ml10
if create_ml10_config:

    ml10_test_tasks = ["door-close-v2", "drawer-open-v2", "lever-pull-v2", "shelf-place-v2", "sweep-v2"]
    ml10_train_tasks = ["basketball-v2", "button-press-v2", "dial-turn-v2", "drawer-close-v2", "peg-insert-side-v2",
                        "pick-place-v2", "push-v2", "reach-v2", "sweep-into-v2", "window-open-v2"]
    ml10_train_task_numbers = [env_mapping[cur] for cur in ml10_train_tasks if cur in env_mapping.keys()]
    ml10_test_tasks_numbers = [env_mapping[cur] for cur in ml10_test_tasks if cur in env_mapping.keys()]

    task_config_ml10 = {}
    task_config_ml10["env"] = "ml10"
    task_config_ml10["total_tasks"] = 10,
    task_config_ml10["train_tasks"] = ml10_train_task_numbers
    task_config_ml10["test_tasks"] = ml10_test_tasks_numbers
    with open(os.path.join(out_dir, 'metaworld_ml10.json'), 'w') as config_json_ml10:
        json.dump(task_config_ml10, config_json_ml10)


print("Done!!")