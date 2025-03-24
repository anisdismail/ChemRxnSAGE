import logging
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions
from scipy.spatial.distance import cdist, jensenshannon, pdist, squareform
from vendi_score import vendi

from utils import get_atoms, get_PO_bonds, rxn_to_chain_ids, rxn_to_ring_ids

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def similarity(fps, centroids, metric):
    if len(fps) > 0:
        dists = cdist(fps, centroids, metric)
        return 1 - np.mean(np.min(dists, axis=1)), 1 - np.min(dists, axis=1)
    return 0, 0


"""
Filter 0: Logical Usage of Elements in Product
Make sure all elements used in product comes from reactants
"""


def filter_1(rxn):
    # Split the reaction string into individual molecules
    mols = re.split(r"[>>|.]", rxn)

    # Separate reactants and products
    reactants = [mol for mol in mols[:-1] if mol]
    products = [mols[-1]]

    # Initialize lists to store atom types from reactants and products
    reactant_atoms = set()
    product_atoms = set()
    # Process reactants
    for mol in reactants:
        if mol:
            # Convert the molecule to a canonical SMILES string without isomeric information
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(smiles)

            # Extract atoms from the molecule
            reactant_atoms.update(set(get_atoms(mol)))

    # Process products
    for mol in products:
        if mol:
            # Convert the molecule to a canonical SMILES string without isomeric information
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(smiles)

            # Extract atoms from the molecule
            product_atoms.update(set(get_atoms(mol)))

    # Check if all product atoms are present in the reactant atoms
    return product_atoms.issubset(reactant_atoms)


""" 
Filter 2: Illogical Ring Operations
• D) Addition of an atom into a ring
• E) Replacement of an atom in a ring (ring stays at the same size)
• F) Transformations that lead to addition/missing atom(s) in a ring
• F) Transformations that lead to addition/missing carbon in a ring

"""


def filter_2(rxn, thresh=0):
    # logging.info(rxn)
    all_react_ids, all_react_systems, prod_ids, prod_systems, react, react_dict = (
        rxn_to_ring_ids(rxn)
    )
    deleted = {}
    new_rings = []

    for ids in all_react_ids:
        for id in ids:
            if id not in deleted:
                deleted[id] = False

    # make sure all rings in product exist in reactants and new rings will be processed later
    for group in prod_systems:
        if sorted(group) in [sorted(el) for el in all_react_systems]:
            index = [sorted(el) for el in all_react_systems].index(sorted(group))
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
            ring = [
                el.replace("+", "").replace("-", "").replace("[", "").replace("]", "")
                for el in ring
            ]
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


def filter_3(rxn, thresh=0):
    _, all_react_systems, _, prod_systems = rxn_to_chain_ids(rxn)
    new_chains = []
    # make sure all chains in product exist in reactants
    # else new chains will be checked later
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
        new_chain.sort()
        for i in range(len(all_react_systems) - 1):
            for j, el2 in enumerate(all_react_systems):
                if i != j:
                    el1 = all_react_systems[i]
                    temp = all_react_systems[i] + all_react_systems[j]
                    temp_list = list(temp)
                    temp_list.sort()
                    if temp_list == new_chain:
                        all_react_systems.remove(el1)
                        all_react_systems.remove(el2)
                        found = True
                        break
            if found:
                break
        # No valid pairing found
        if not found:
            return False

    return True
    # caveat: chain to ring formation is not allowed

    # TODO: faster graph edit path calculation to make sure no chain to ring formation is happening
    # else:
    #  trans=return_Chain_trans(rxn,all_react_ids)
    #  breakage_perc=trans["Chain_Breakage"]/trans["Nb_Paths"]
    #  return True if breakage_perc<=thresh and prod_systems==all_react_systems else False


"""
Filter 5: P-O Bond Cleavage
"""

"""
def filter_5(rxn, thresh=0):
    mols = re.split(r"[>>|.]", rxn)
    react = [mol for mol in mols[:-1] if mol]
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


def filter_4(rxn, thresh=0):
    mols = re.split(r"[>>|.]", rxn)
    react = [mol for mol in mols[:-1] if mol]
    prod = mols[-1]
    if not prod:
        return False

    react_PO_bonds_single, react_PO_bonds_double = [], []
    for mol in react:
        if mol:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(smiles)
            PO_bonds = get_PO_bonds(mol)
            react_PO_bonds_single += PO_bonds["single"]
            react_PO_bonds_double += PO_bonds["double"]

    prod = Chem.MolToSmiles(Chem.MolFromSmiles(prod), isomericSmiles=False)
    prod = Chem.MolFromSmarts(prod)
    PO_bonds = get_PO_bonds(prod)
    prod_PO_bonds_single = PO_bonds["single"]
    prod_PO_bonds_double = PO_bonds["double"]

    total_react_PO_bonds = len(react_PO_bonds_single) + len(react_PO_bonds_double)
    total_prod_PO_bonds = len(prod_PO_bonds_single) + len(prod_PO_bonds_double)

    if (total_react_PO_bonds == total_prod_PO_bonds) and (
        len(prod_PO_bonds_double) >= len(react_PO_bonds_double)
    ):
        # caveat: make sure no P-O cleavage is happening then P-O bond forming (nb will stay same)
        # caveat: what about when the cleavage remove the whole structure from the product
        return True
    else:
        return False

    # TODO: check if P-O substructures left the product as a whole while no cleavage
    # check whether P or O are in the product still or not


# TODO: faster graph edit path calculation to make sure no P-O cleavage is happening then P-O bond forming same time
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
    # Check if the reaction format is valid using a regular expression
    if not re.fullmatch(r"[^>]+[>]{2}[^>]+", rxn):
        return False

    # Split the reaction string into individual molecules
    mols = re.split(r"[>>|.]", rxn)
    mols = [mol for mol in mols if mol]

    # Validate each molecule in the reaction
    for mol in mols:
        rdkit_mol = Chem.MolFromSmiles(mol)
        if not rdkit_mol:
            return False

        # Convert to SMILES and back to SMARTS ensure validity
        smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)
        if not Chem.MolFromSmarts(smiles):
            return False

    # Validate the overall reaction format
    try:
        rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
    except Exception as e:
        logging.info("Error in reaction format:", e)
        return False

    return True


"""
Reaction Variety Metric
"""


def JSS_with_train(rxn_pred):
    if len(rxn_pred) > 0:
        train_data_dist = {
            10: 0.004521721751729995,
            1: 0.30223088261010767,
            2: 0.23810237577756127,
            3: 0.11269329735941443,
            4: 0.017986959454395563,
            5: 0.012990581828174573,
            6: 0.16695395838017438,
            7: 0.09160858377676184,
            8: 0.01626320917334932,
            9: 0.03664842988833096,
        }
        train_dist_arr = [train_data_dist[i] for i in range(1, 11)]
        rxn_pred = [10 if x == 0 else x for x in rxn_pred]

        rxn_pred_dist = pd.Series(rxn_pred).value_counts(normalize=True).to_dict()
        if len(rxn_pred_dist) < 10:
            for i in range(1, 11):
                if i not in rxn_pred_dist.keys():
                    rxn_pred_dist[i] = 0
        rxn_pred_dist_arr = [rxn_pred_dist[i] for i in range(1, 11)]
        return 1 - jensenshannon(train_dist_arr, rxn_pred_dist_arr)
    else:
        return 0


"""
Percentage of Novel Reactions in Dataset
"""


def novelty_percentage(df_gen_valid, df_ref):
    ref_set = set(df_ref["Reactant_Product_Frozenset"])

    # Check if each frozenset in df_gen_valid exists in max_200_set
    df_gen_valid["Exists_In_Max_200"] = df_gen_valid[
        "Reactant_Product_Frozenset"
    ].apply(lambda x: x in ref_set)
    perc_exac_matches = df_gen_valid["Exists_In_Max_200"].sum() / len(
        df_gen_valid["Exists_In_Max_200"]
    )
    return 1 - perc_exac_matches


"""
Percentage of Unique Reactions in Generated Dataset
"""


def unique_percentage(df_gen_valid):
    return df_gen_valid["Reactant_Product_Frozenset"].nunique() / len(
        df_gen_valid["Exists_In_Max_200"]
    )


"""
Dataset Diversity using Vendi score
from https://github.com/vertaix/Vendi-Score
We recommend pairing sample quality metrics with a Vendi score of small
order (q ∈ [0.1,0.5]) for diversity and the Vendi score of infinite order for
duplication and memorization.
"""


def calculate_dataset_diversity(gen_fingerprints):
    X_sims = 1 - squareform(pdist(gen_fingerprints, metric="jaccard"))
    upper_tri_indices = np.triu_indices_from(X_sims, k=1)
    average_inter_similarity = np.mean(X_sims[upper_tri_indices])
    vendi_score_k = vendi.score_K(X_sims)
    vendi_score_k_inf = vendi.score_K(X_sims, q="inf")
    vendi_score_k_small = vendi.score_K(X_sims, q=0.1)

    return (
        vendi_score_k,
        vendi_score_k_inf,
        vendi_score_k_small,
        1 - average_inter_similarity,
    )


"""
Vendi for Score for Every Predicted Class
"""


def calculate_diversity_per_class(rxn_pred, gen_fingerprints):
    rxn_pred_arr = np.array(rxn_pred)
    results = []
    for i in np.unique(rxn_pred_arr):
        # Select fingerprints corresponding to the current class
        select_fingerprints = gen_fingerprints[np.where(rxn_pred_arr == i)]

        # Calculate diversity for the selected fingerprints
        diversity = calculate_dataset_diversity(select_fingerprints)

        # Append the results (label, count, first diversity score, last diversity score)
        results.append(
            {
                "label": i,
                "count": len(select_fingerprints),
                "VS": diversity[0],
                "AvgInterSim": diversity[-1],
            }
        )
    results_df = pd.DataFrame(results)
    results_df["VS_norm"] = results_df["VS"] / results_df["count"]
    return results_df
