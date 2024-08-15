import re
import pandas as pd
import numpy as np
import logging

from vendi_score import vendi
from rdkit.Chem import rdChemReactions
from rdkit import Chem
from rdkit import RDLogger
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import cdist, jensenshannon, pdist, squareform
from utils import get_atoms, rxn_to_chain_ids, rxn_to_ring_ids, get_PO_bonds

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def similarity(fps, centroids, metric):
    if len(fps) > 0:
        dists = cdist(fps, centroids, metric)
        return 1-np.mean(np.min(dists, axis=1)), 1-np.min(dists, axis=1)
    return 0, 0


"""
Filter 0: Logical Usage of Elements in Product
Make sure all elements used in product comes from reactants
"""


def filter_0(rxn):
    # Split the reaction string into individual molecules
    mols = re.split(r"[>>|.]", rxn)

    # Separate reactants and products
    reactants = [mol for mol in mols[:-1] if mol]
    products = [mols[-1]]

    # Initialize lists to store atom types from reactants and products
    reactant_atoms = []
    product_atoms = []
    # Process reactants
    for mol in reactants:
        if mol:
            # Convert the molecule to a canonical SMILES string without isomeric information
            smiles = Chem.MolToSmiles(
                Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(smiles)

            # Extract atoms from the molecule
            atoms = get_atoms(mol)
            reactant_atoms.extend(atoms)

    # Process products
    for mol in products:
        if mol:
            # Convert the molecule to a canonical SMILES string without isomeric information
            smiles = Chem.MolToSmiles(
                Chem.MolFromSmiles(mol), isomericSmiles=False)
            mol = Chem.MolFromSmarts(smiles)

            # Extract atoms from the molecule
            atoms = get_atoms(mol)
            product_atoms.extend(atoms)

    # Check if all product atoms are present in the reactant atoms
    return set(product_atoms).issubset(set(reactant_atoms))


""" 
Filter 2: Illogical Ring Operations
• D) Addition of an atom into a ring
• E) Replacement of an atom in a ring (ring stays at the same size)
• F) Transformations that lead to addition/missing atom(s) in a ring
• F) Transformations that lead to addition/missing carbon in a ring

"""


def filter_2(rxn, thresh=0):
    # logging.info(rxn)
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

        # Convert to SMILES and back to ensure validity
        smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)
        if not Chem.MolFromSmarts(smiles):
            return False

    # Validate the overall reaction format
    try:
        rdChemReactions.ReactionFromSmarts(rxn, useSmiles=False)
    except Exception as e:
        logging.info("Error in reaction format:", e)
        return False

    return True


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


"""
Percentage of Exact Matches in Dataset
"""


def exact_matches_percentage(df_gen_valid, df_ref):
    ref_set = set(df_ref['Reactant_Product_Frozenset'])

    # Check if each frozenset in df_gen_valid exists in max_200_set
    df_gen_valid['Exists_In_Max_200'] = df_gen_valid['Reactant_Product_Frozenset'].apply(
        lambda x: x in ref_set)
    perc_exac_matches = df_gen_valid['Exists_In_Max_200'].sum(
    ) / len(df_gen_valid['Exists_In_Max_200'])
    return perc_exac_matches


"""
Percentage of Duplicate Reactions in Generated Dataset
"""


def percentage_duplicates(df_gen_valid):
   # Calculate the value counts of 'Reactant_Product_Frozenset' in the dataframe
    frozenset_counts = df_gen_valid['Reactant_Product_Frozenset'].value_counts(
    )
    # Filter for frozensets that appear more than once
    frequent_frozensets = frozenset_counts[frozenset_counts > 1]

    # Calculate the sum of these frequent frozensets
    sum_frequent_frozensets = frequent_frozensets.sum()

    # Calculate the proportion of frequent frozensets to total rows
    perc_duplicates = sum_frequent_frozensets / \
        len(df_gen_valid['Exists_In_Max_200'])

    return perc_duplicates


"""
Dataset Diversity using Vendi score
from https://github.com/vertaix/Vendi-Score
We recommend pairing sample quality metrics with a Vendi score of small
order (q ∈ [0.1,0.5]) for diversity and the Vendi score of infinite order for
duplication and memorization.
"""


def calculate_dataset_diversity(gen_fingerprints):

    X_sims = 1-squareform(pdist(gen_fingerprints, metric='jaccard'))
    upper_tri_indices = np.triu_indices_from(X_sims, k=1)
    average_inter_similarity = np.mean(X_sims[upper_tri_indices])
    vendi_score_k = vendi.score_K(X_sims)
    vendi_score_k_inf = vendi.score_K(X_sims, q="inf")
    vendi_score_k_small = vendi.score_K(X_sims, q=0.1)

    return vendi_score_k, vendi_score_k_inf, vendi_score_k_small, average_inter_similarity


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
        results.append({
            'label': i,
            'count': len(select_fingerprints),
            'VS': diversity[0],
            'AvgInterSim': diversity[-1]
        })
    results_df = pd.DataFrame(results)
    results_df["VS_norm"] = results_df["VS"]/results_df["count"]
    return results_df
