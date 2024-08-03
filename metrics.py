import re
import pandas as pd
import numpy as np

from rdkit.Chem import rdChemReactions
from rdkit import Chem
from rdkit import RDLogger

from scipy.spatial.distance import cdist, jensenshannon

from utils import get_atoms, rxn_to_chain_ids, rxn_to_ring_ids, get_PO_bonds

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def similarity(fps, centroids, metric):
    if len(fps) > 0:
        sims = cdist(fps, centroids, metric)
        return 1-np.mean(np.min(sims, axis=1)), np.min(sims, axis=1)
    return 0, 0


"""
Filter 0: Logical Usage of Elements in Product
Make sure all elements used in product comes from reactants
"""


def filter_0(rxn):
    mols = re.split(r"[>>|.]", rxn)
    react = [mol for mol in mols[:-1] if len(mol) >= 1]
    prod = [mols[-1]]

    all_react_systems = []
    prod_systems = []

    for mol in react:
        if mol:
            mol = Chem.MolToSmiles(
                Chem.MolFromSmiles(mol), isomericSmiles=False)
            # if not Chem.MolFromSmarts(mol):
            # print(mol)
            mol = Chem.MolFromSmarts(mol)
            atoms = get_atoms(mol)
            all_react_systems += atoms

    for mol in prod:
        if mol:
            mol = Chem.MolToSmiles(
                Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(mol)
            atoms = get_atoms(mol)
            prod_systems += atoms

    return set(set(prod_systems)).issubset(set(all_react_systems))


""" 
Filter 2: Illogical Ring Operations
• D) Addition of an atom into a ring
• E) Replacement of an atom in a ring (ring stays at the same size)
• F) Transformations that lead to addition/missing atom(s) in a ring
• F) Transformations that lead to addition/missing carbon in a ring

"""


def filter_2(rxn, thresh=0):
    # print(rxn)
    all_react_ids, all_react_systems, prod_ids, prod_systems, react, react_dict = rxn_to_ring_ids(
        rxn)
    deleted = {}
    new_rings = []

    for ids in all_react_ids:
        for id in ids:
            if not id in deleted:
                deleted[id] = False

    # make sure all rings in product exist in reactants and new rings will be processed later
    for group in prod_systems:
        if sorted(group) in [sorted(el) for el in all_react_systems]:
            index = [sorted(el)
                     for el in all_react_systems].index(sorted(group))
            id_group = all_react_ids[index]
            for id in id_group:
                if not deleted[id]:
                    for atom in react_dict[id]:
                        react.remove(atom)
                    # mark atoms as used/ no longer candidates for new rings formation
                    deleted[id] = True
            del all_react_ids[index]
            del all_react_systems[index]
        else:
            new_rings.append(group)

    # creating a ring is fine in product - pericyclic/click chemistry
    # TODO: when forming new rings, it is okay for carbon to become aromatic (C>>c)
    if len(new_rings) == 0:
        return True
    else:
        for ring in new_rings:
            ring = [el.replace(
                "+", "").replace("-", "").replace("[", "").replace("]", "") for el in ring]
            for atom in ring:
                if atom in react:
                    react.remove(atom)
                else:
                    return False
        return True

    # TODO: faster graph edit path calculation to make sure no ring cleavage is happening
    # trans=return_Ring_trans(rxn)
    # breakage_perc=trans["Ring_Breakage"]/trans["Nb_Paths"]
    # return True if breakage_perc<=thresh else False


"""
  Filter 4: Illogical Chain Operations
• G) Transformations that lead to addition/missing atom(s) in a chain
• G) Transformations that lead to addition/missing carbon in a chain
  """


def filter_4(rxn, thresh=0):

    all_react_ids, all_react_systems, prod_ids, prod_systems = rxn_to_chain_ids(
        rxn)
    new_chains = []
    # make sure all chains in product exist in reactants, else new chains will be checked later
    for chain in prod_systems:
        if chain in all_react_systems:
            all_react_systems.remove(chain)
        else:
            new_chains.append(chain)

    if len(new_chains) == 0:
        return True
    # creating a chain is fine in product
    for new_chain in new_chains:
        found = False
        new_chain = list(new_chain)
        # Sort the given string
        new_chain.sort()
        # Select two strings at a time from given vector
        for i in range(len(all_react_systems)-1):
            for j, el2 in enumerate(all_react_systems):
                if i != j:
                    el1 = all_react_systems[i]
                    # Get the concatenated string
                    temp = all_react_systems[i] + all_react_systems[j]
                    # Sort the resultant string
                    temp_list = list(temp)
                    temp_list.sort()
                    if (temp_list == new_chain):
                        all_react_systems.remove(el1)
                        all_react_systems.remove(el2)
                        found = True
                        break
            if found:
                break
        # No valid pair found
        if not found:
            return False
    return True

    # TODO: faster graph edit path calculation to make sure no ring cleavage is happening
    # else:
    #  trans=return_Chain_trans(rxn,all_react_ids)
    #  breakage_perc=trans["Chain_Breakage"]/trans["Nb_Paths"]
    #  return True if breakage_perc<=thresh and prod_systems==all_react_systems else False


"""
Filter 5: P-O Bond Cleavage
"""


def filter_5(rxn, thresh=0):
    mols = re.split(r"[>>|.]", rxn)
    react = [mol for mol in mols[:-1] if len(mol) >= 1]
    prod = [mols[-1]]

    all_react_systems = []
    for mol in react:
        if mol:
            mol = Chem.MolFromSmiles(mol)
            all_react_systems += get_PO_bonds(mol)
    prod_systems = []
    for mol in prod:
        if mol:
            mol = Chem.MolFromSmiles(mol)
            prod_systems += get_PO_bonds(mol)

    if (len(prod_systems) != len(all_react_systems)):
        return False
    return True
   # TODO: faster graph edit path calculation to make sure no ring cleavage is happening
   # else:
   #   trans=return_PO_trans(rxn)
   #   PO_breakage_perc=trans["PO_Breakage"]/trans["Nb_Paths"]
   #   return True if PO_breakage_perc<=thresh else False


"""
Validity Metric
Next, let's create a metric that measures the format validity of each chemical reaction: 
Each molecule is smiles valid
The reaction follows the following pattern : ....>>....
"""


def is_valid_rxn(rxn):
    if re.fullmatch(r"[^>]+[>]{2}[^>]+", rxn) is None:
        return False
    mols = re.split(r"[>>|.]", rxn)
    mols = [i for i in mols if i]
    valid = True
    for mol in mols:
        mol = Chem.MolFromSmiles(mol)
        valid = valid and mol is not None
    if valid:
        try:
            Chem.rdChemReactions.ReactionFromSmarts(
                rxn, useSmiles=True)
        except:
            valid = valid and False
            print("error in reaction format")
        return valid
    return False


"""
Reaction Variety Metric
"""


def JSD_with_train(rxn_pred):
    if len(rxn_pred) > 0:
        train_data_dist = {10: 0.004521721751729995, 1: 0.30223088261010767, 2: 0.23810237577756127, 3: 0.11269329735941443,
                           4: 0.017986959454395563, 5: 0.012990581828174573, 6: 0.16695395838017438, 7: 0.09160858377676184,
                           8: 0.01626320917334932, 9: 0.03664842988833096}
        train_dist_arr = [train_data_dist[i] for i in range(1, 11)]
        rxn_pred = [10 if x == 0 else x for x in rxn_pred]

        rxn_pred_dist = pd.Series(rxn_pred).value_counts(
            normalize=True).to_dict()
        if len(rxn_pred_dist) < 10:
            for i in range(1, 11):
                if i not in rxn_pred_dist.keys():
                    rxn_pred_dist[i] = 0
        rxn_pred_dist_arr = [rxn_pred_dist[i] for i in range(1, 11)]
        return 1-(jensenshannon(train_dist_arr, rxn_pred_dist_arr)**2)
    else:
        return 0
