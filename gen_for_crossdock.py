# conda activate prefixmol
# this will generate 100 new ligands for each pair in the crossdock dataset
import warnings
warnings.filterwarnings('ignore')
import os
import os.path as osp
from glob import glob
import torch
import torch.utils.data
import os.path as osp
from parser_1 import create_parser
from methods.MolDesign import MolDesign
from utils.main_utils import print_log, output_namespace, check_dir, load_config
from utils.pdb2data import SinglePDBDataset, read_sdf, write_sdf, fix_state_dict, generate_pos
from utils.load_data import DataLoader, collate_fn_sparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_scatter import scatter_sum
from rdkit import Chem
from utils.mol_utils import get_metric, get_metric_mol
import argparse
from rdkit import RDLogger
import numpy as np
import shutil
from tqdm import tqdm
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)

if __name__ == '__main__':
# torch.distributed.init_process_group(backend='nccl')
    args = create_parser()
    config = args.__dict__
    default_params = load_config(osp.join('./configs', args.method + '.py' if args.config_file is None else args.config_file))
    config.update(default_params)
    print(config)

    method = MolDesign(args, args.device, 1)
    # Load the state_dict from your checkpoint
    model_path = './ckpt/checkpoint.pth'
    fixed_state_dict = fix_state_dict(torch.load(model_path))

    # Load the fixed state_dict into your model
    method.model.load_state_dict(fixed_state_dict, strict=False)

    targets = glob('./data/crossdock_pocket_test/*')

    for target in tqdm(targets):
        try:
            full_protein = min(glob(osp.join(target, '*.pdb')), key=len)
            protein_path = max(glob(osp.join(target, '*.pdb')), key=len)
            ligand_path = glob(osp.join(target, '*.sdf'))[0]
            save_base = './results/crossdock'
            ligand_dir = ligand_path.split('/')[-2]
            ligand_name = ligand_path.split('/')[-1].split('.')[0]
            gen_ligand_name = ligand_name + '_prefixmol.sdf'
            save_path = osp.join(save_base, ligand_dir)
            SDF_dir = osp.join(save_path, 'SDF')
            ori_dir = osp.join(save_path, 'ori')
            os.makedirs(ori_dir, exist_ok=True)
            shutil.copy(ligand_path, osp.join(ori_dir,'0.sdf'))
            if osp.exists(osp.join(save_path, gen_ligand_name)):
                continue
            os.makedirs(SDF_dir, exist_ok=True)
            
            shutil.copy(ligand_path, save_path)
            shutil.copy(full_protein, save_path)

            complex_pairs = [(protein_path, ligand_path)]
            dataset = SinglePDBDataset(complex_pairs)
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_sparse, pin_memory=True)

            total_gen = 0
            all_mols = []
            cnt = 0
            while len(all_mols) < 100:
                cnt+=1
                if cnt > 100:
                    print('Bug Existed!')
                    break

                ori_mol = read_sdf(ligand_path)[0]
                ori_opt_config = get_metric_mol(ori_mol)
                gen_opt_config = {'vina': 0 + np.random.randn()*0.1, 
                                'qed': ori_opt_config['qed'] + np.random.randn()*0.1, 
                                'sa': ori_opt_config['sa'] + np.random.randn()*0.1, 
                                'lipinski': ori_opt_config['lipinski'] + np.random.randn()*0.1, 
                                'logp': ori_opt_config['logp'] + np.random.randn()*0.1}


                metrics = []
                for batch in data_loader:
                    protein_file_list = batch[-3]
                    smiles_list = batch[-2]
                    metrics_list = batch[-1]
                    
                    batch = [one.to(args.device) for one in batch[:-3]]
                    
                    start_idx, protein_feature, protein_pos,  protein_batch_id, protein_edge_idx = batch 
                    device = protein_pos.device
                    N_atoms_protein = scatter_sum(torch.ones_like(protein_batch_id), protein_batch_id)
                    shift_protein = torch.cumsum(torch.cat([torch.zeros([1], device=device).long(), N_atoms_protein]), dim=-1)[:-1]
                    
                    idx_context = start_idx+shift_protein 

                    with torch.no_grad():
                        gen_mols = []
                        pred_smiles = method.model(batch, smiles_list, metrics_list, protein_file_list, mode='test', opt_config=gen_opt_config)
                        # clean the pred_smiles to split the pdb file
                        WordFilter = set(['pdb'])
                        pred_smiles_list = [smiles for smiles in pred_smiles if not any(word in smiles for word in WordFilter)]
                        gt_vina = [0 for i in metrics_list] # i['vina'] for vina version
                        for b in range(len(pred_smiles_list)):

                            mol = Chem.MolFromSmiles(pred_smiles_list[b])
                            gen_mols.append(mol)
                        gen_mols = [mol for mol in gen_mols if mol is not None]
                        gen_geom_mols = []
                        for mol in gen_mols:
                            try:
                                gen_geom_mols.append(generate_pos(mol))
                            except:
                                pass
                        all_mols.extend(gen_geom_mols)

            for i, mol in enumerate(all_mols):
                write_sdf([mol], osp.join(SDF_dir, f'{i}.sdf'), voice=False)

            write_sdf(all_mols, osp.join(save_path, gen_ligand_name), voice=False)
        
        except Exception as e:
            print(e)
            continue