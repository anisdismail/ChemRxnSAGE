import re
import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem


def load_rxn_classifier(config):
    rxn_classifier = torch.load(os.path.join(
        config["project_dir"], "final_nn_classifier.pth"))
    rxn_classifier.eval()
    return rxn_classifier


def predict_rxn_type(model, gen_fingerprints, batch_size=64):
    rxn_pred = []

    # Convert all fingerprints to a single Tensor
    fingerprint_tensor = torch.Tensor(gen_fingerprints).float().cuda()

    # Split fingerprints into batches
    num_samples = len(fingerprint_tensor)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_fingerprints = fingerprint_tensor[start_idx:end_idx]

        # Perform predictions for the batch
        outputs = model(batch_fingerprints)
        y_pred_softmax = torch.log_softmax(outputs, dim=0)
        y_pred_tags = torch.argmax(y_pred_softmax, dim=0)

        rxn_pred.extend(y_pred_tags.cpu().numpy().tolist())

    return rxn_pred


def get_class_10_stats(rxn_pred):
    rxn_pred_dist = dict(pd.Series(rxn_pred).value_counts(normalize=True))
    # class 10 correspond to key 0
    return rxn_pred_dist[0] if 0 in rxn_pred_dist.keys() else 0


def get_atoms(mol):
    atoms = [atom.GetSmarts().upper() for atom in mol.GetAtoms()]
    return atoms


def node_match(node1, node2):
    return node1['smarts'] == node2['smarts']


def get_rings(mol):
    ri = mol.GetRingInfo()
    ids = [ring for ring in ri.AtomRings()]
    smiles = [[mol.GetAtomWithIdx(i).GetSmarts()
               for i in ring] for ring in ri.AtomRings()]
    return ids, smiles


def is_ring_edge(edge, ids):
    if edge is not None:
        for ring in ids:
            if edge[0][2] in ring and edge[1][2] in ring:
                return True
        return False


def rxn_to_ring_ids(rxn):
    mols = re.split(r"[>>|.]", rxn)
    react = [mol for mol in mols[:-1] if len(mol) >= 1]
    prod = [mols[-1]]

    all_react_systems = []
    all_react_ids = []
    prod_systems = []
    prod_ids = []
    all_react = ''
    react_id_2_smiles = {}
    for mol in react:

        if mol:

            mol = Chem.MolToSmiles(
                Chem.MolFromSmiles(mol), isomericSmiles=False)
            all_react += mol
            mol = Chem.MolFromSmarts(mol)
            ids, smiles = get_rings(mol)
            all_react_systems += smiles
            all_react_ids += ids

            for atom in mol.GetAtoms():
                id = atom.GetIdx()
                if not id in react_id_2_smiles:
                    react_id_2_smiles[id] = atom.GetSmarts()
    for mol in prod:
        if mol:
            mol = Chem.MolToSmiles(
                Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(mol)
            ids, smiles = get_rings(mol)
            prod_systems += smiles
            prod_ids += ids

    return all_react_ids, all_react_systems, prod_ids, prod_systems, list(all_react), react_id_2_smiles


def get_chains(mol):
    # get unique chains with no subchains
    hit_ats = []
    hit_ats_smiles = []
    for nb_c_chain in range(10, 0, -1):
        patt = Chem.MolFromSmiles('C'*nb_c_chain)
        match = list(mol.GetSubstructMatches(patt))
        filtered = match.copy()
        for chain in hit_ats:
            for el in match:
                if set(el).issubset(chain) and el in filtered:
                    filtered.remove(el)
        hit_ats += filtered
        hit_ats_smiles += ["".join(sorted([mol.GetAtomWithIdx(j).GetSmarts()
                                   for j in i])) for i in filtered]
    return hit_ats, hit_ats_smiles


def rxn_to_chain_ids(rxn):
    mols = re.split(r"[>>|.]", rxn)
    react = [mol for mol in mols[:-1] if len(mol) >= 1]
    prod = [mols[-1]]

    all_react_systems = []
    all_react_ids = []
    prod_systems = []
    prod_ids = []

    for mol in react:
        if mol:
            mol = Chem.MolToSmiles(
                Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(mol)
            ids, smiles = get_chains(mol)
            all_react_systems += smiles
            all_react_ids += ids
    for mol in prod:
        if mol:
            mol = Chem.MolToSmiles(
                Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(mol)
            ids, smiles = get_chains(mol)
            prod_systems += smiles
            prod_ids += ids
    return all_react_ids, all_react_systems, prod_ids, prod_systems


def is_chain_edge(edge, ids):
    if edge is not None:
        for chain in ids:
            if edge[0][2] in chain and edge[1][2] in chain:
                return True
        return False


def get_PO_bonds(mol):
    patt = Chem.MolFromSmarts('PO')
    hit_ats = list(mol.GetSubstructMatches(patt))
    return hit_ats


def is_PO_edge(edge):
    if edge is not None:
        if edge[0][1] == "P" and edge[1][1] == "O":
            return True
        if edge[0][1] == "O" and edge[1][1] == "P":
            return True
    return False


def convert_fingerprint(row):
    array = np.zeros((0,), dtype=np.int64)
    Chem.DataStructs.ConvertToNumpyArray(row["fingerprint"], array)
    return array
