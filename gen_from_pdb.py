# python gen_from_pdb.py --protein_path ./protein.pdb --ligand_path ./ligand.sdf --save_base ./results --num_gen 100
# this will use the qed, sa, and logp (adding noise) of the original ligand to generate 100 new ligands
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

if __name__ == 'main':
# torch.distributed.init_process_group(backend='nccl')
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument("--local_rank", default=0, type=int, help="Used for DDP, local rank means the process number of each machine")
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='pocket2smiles_verify_metrics', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=8, type=int)
    parser.add_argument('--seed', default=111, type=int)
    # CATH

    # dataset parameters
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--given_rc', default=0, type=int)
    parser.add_argument('--use_motif_action', default=0, type=int)
    parser.add_argument('--use_motif_feature', default=0, type=int)
    parser.add_argument('--use_hierachical_action', default=0, type=int)

    # method parameters
    parser.add_argument('--method', default='MolDesign', choices=["MolDesign"])
    parser.add_argument('--config_file', '-c', default=None, type=str)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--sparse', default=1, type=int)


    # Training parameters
    parser.add_argument('--epoch', default=50, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=100, type=int)
    
    parser.add_argument('--protein_path', default='./protein.pdb', type=str)
    parser.add_argument('--ligand_path', default='./ligand.sdf', type=str)
    parser.add_argument('--save_base', default='./results', type=str)
    parser.add_argument('--num_gen', default=100, type=int)
    args = parser.parse_args()

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


    protein_path = args.protein_path
    ligand_path = args.ligand_path
    save_base = args.save_base

    ligand_name = ligand_path.split('/')[-1].split('.')[0]
    protein_name = protein_path.split('/')[-1].split('.')[0]
    save_dir = protein_name + '_' + ligand_name
    gen_ligand_name = ligand_name + '_prefixmol.sdf'
    save_path = osp.join(save_base, save_dir)
    SDF_dir = osp.join(save_path, 'SDF')
    ori_dir = osp.join(save_path, 'ori')
    os.makedirs(ori_dir, exist_ok=True)
    shutil.copy(ligand_path, osp.join(ori_dir,'0.sdf'))
    os.makedirs(SDF_dir, exist_ok=True)    
    shutil.copy(ligand_path, save_path)


    complex_pairs = [(protein_path, ligand_path)]
    dataset = SinglePDBDataset(complex_pairs)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_sparse, pin_memory=True)

    total_gen = 0
    all_mols = []
    while len(all_mols) < args.num_gen:
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
                all_mols.extend(gen_mols)

    for i, mol in enumerate(all_mols):
        write_sdf([mol], osp.join(SDF_dir, f'{i}.sdf'), voice=False)

    write_sdf(all_mols, osp.join(save_path, gen_ligand_name), voice=False)

