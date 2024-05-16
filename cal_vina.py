import os
import pickle
from rdkit import Chem
import argparse
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.docking import QVinaDockingTask

def cut_list(lists, cut_len):
    res_data = []
    if len(lists) > cut_len:
        for i in range(int(len(lists) / cut_len)):
            cut_a = lists[cut_len * i:cut_len * (i + 1)]
            res_data.append(cut_a)

        last_data = lists[int(len(lists) / cut_len) * cut_len:]
        if last_data:
            res_data.append(last_data)
    else:
        res_data.append(lists)
    return res_data

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--idx', type=int, default=1) 
args = arg_parser.parse_args()
print('now idx = ', args.idx)

with open('data/crossdocked_pocket10/index.pkl', 'rb') as f:
    index = pickle.load(f)

tasks = []
raw_path = 'data/crossdocked_pocket10'
for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(index):
    if pocket_fn is None: continue
    tasks.append((i, pocket_fn, ligand_fn))

if 10000 * args.idx > len(tasks):
    subtasks = tasks[10000*(args.idx-1) :]
else:
    subtasks = tasks[10000*(args.idx-1) : 10000*args.idx]

results=[]
for (i, pocket_fn, ligand_fn) in subtasks:
    # data_list = handle_per_task(i, pocket_fn, ligand_fn, raw_path=self.raw_path)
    pocket_dict = PDBProtein(os.path.join(raw_path, pocket_fn)).to_dict_atom()
    #try:
    ligand_dict = parse_sdf_file(os.path.join(raw_path, ligand_fn))
    # except:
    #     #print('invalid_smi')
    #     continue
    rdmol = ligand_dict['rdmol']
    smi = Chem.MolToSmiles(rdmol)
    task_id=(pocket_fn+"_SEP_"+ligand_fn).replace("/", "__")
    #try:
    vina_task = QVinaDockingTask.from_generated_data(pocket_fn , rdmol, task_id=task_id)
    docking_results = vina_task.run_sync(center = None, patience=None)
    vina = docking_results[0]['affinity']        
    # except:
    #     continue

    results.append([pocket_fn, smi, vina])
    if (i+1)%100==0:
        print('processed: ',i+1)
# save
with open(f'vina_results_{args.idx}.txt','w') as f:
    for p in results:
        f.write('%s,%s,%.2f\n'%(p[0],p[1],p[2]))
