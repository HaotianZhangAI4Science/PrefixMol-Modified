from utils.mol_utils import get_metric
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.data import torchify_dict, ProteinLigandData
from utils.datasets.pl import custom_preprocess
from utils.datasets.moldesign_dataset import RefineData, LigandCountNeighbors, FeaturizeProteinAtom, FeaturizeLigandAtom, Compose
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_pos(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    mol = Chem.RemoveHs(mol)
    return mol

def fix_state_dict(state_dict):
    """Remove the 'module.' prefix from the start of each state_dict key."""
    fixed_state_dict = {key[7:] if key.startswith('module.') else key: value
                        for key, value in state_dict.items()}
    return fixed_state_dict

def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file)
    mols_list = [i for i in supp]
    return mols_list

def write_sdf(mol_list,file, voice=False):
    writer = Chem.SDWriter(file)
    mol_cnt = 0
    for i in mol_list:
        try:
            writer.write(i)
            mol_cnt+=1
        except:
            pass
    writer.close()
    if voice: 
        print('Write {} molecules to {}'.format(mol_cnt,file))

def pdb2data(protein_path, ligand_path):
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
            RefineData(),
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer
        ])

    pocket_dict = PDBProtein(protein_path).to_dict_atom()
    ligand_dict = parse_sdf_file(ligand_path)

    rdmol = ligand_dict['rdmol']
    metrics = get_metric(protein_path, rdmol, use_vina=False, task_id=(protein_path+"_SEP_"+ligand_path).replace("/", "__"))
    ligand_dict.update(metrics)

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_path
    data.ligand_filename = ligand_path

    data = custom_preprocess(data, transform)
    return data


class SinglePDBDataset(Dataset):
    def __init__(self, pkt_lig_paris):
        super().__init__()
        self.pkt_lig_paris = pkt_lig_paris

    def __len__(self):
        # Return the number of items in your dataset
        return len(self.pkt_lig_paris)
    
    def __getitem__(self, index):
        try:
            pkt_path = self.pkt_lig_paris[index][0]
            lig_path = self.pkt_lig_paris[index][1]
            data = pdb2data(pkt_path, lig_path)
            if data.ligand_pos.shape[0] > 50:
                return None
            
            metrics = {
                    "vina": 0,
                    "qed": data.ligand_qed,
                    "sa": data.ligand_sa,
                    "lipinski": data.ligand_lipinski,
                    "logp": data.ligand_logp}
            
            protein_mask = (data.protein_is_backbone & data.protein_is_N) | data.protein_is_in_5A
            return data.start_idx, data.protein_atom_feature[protein_mask], data.protein_pos[protein_mask], data.protein_filename, data.protein_one_knn_edge_index, data.ligand_smiles, metrics

        except Exception as e:
            print(e)
            return None
