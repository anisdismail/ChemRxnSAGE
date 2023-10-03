import pandas as pd
import network as nx
import re
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from networkx.algorithms.similarity import optimize_edit_paths, optimal_edit_paths

from utils import rxn_to_ring_ids, node_match, is_chain_edge, is_ring_edge, is_PO_edge


def mol_to_nx(mol, G, order):

    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node((order, atom.GetSmarts(), atom.GetIdx()),
                   smarts=atom.GetSmarts(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
        if i == 0:
            G.add_edge((0, ".", 0,),
                       (order, atom.GetSmarts(), atom.GetIdx()))

    for bond in mol.GetBonds():
        G.add_edge((order, mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSmarts(), bond.GetBeginAtomIdx()),
                   (order, mol.GetAtomWithIdx(bond.GetEndAtomIdx()
                                              ).GetSmarts(), bond.GetEndAtomIdx()),
                   bond_type=bond.GetBondType())
    return G


def rxn_to_graphs(rxn):
    react_graph = nx.Graph()
    prod_graph = nx.Graph()
    mols = re.split(r"[>>|.]", rxn)
    react = [mol for mol in mols[:-1] if len(mol) >= 1]
    # react=[react[0]]
    prod = [mols[-1]]

    react_graph.add_node((0, ".", 0),
                         smarts=".")

    for i, mol in enumerate(react):
        if mol:
            mol = Chem.MolFromSmiles(mol)
            react_graph = mol_to_nx(mol, react_graph, i)

    prod_graph.add_node((0, ".", 0),
                        smarts=".")

    for i, mol in enumerate(prod):
        if mol:
            mol = Chem.MolFromSmiles(mol)
            prod_graph = mol_to_nx(mol, prod_graph, 1)

    return react_graph, prod_graph


def return_ring_trans(rxn):
    all_react_ids, all_react_systems, prod_systems, prod_ids = rxn_to_ring_ids(
        rxn)
    react, prod = rxn_to_graphs(rxn)
    gen = optimize_edit_paths(react, prod, node_match=node_match)
    paths = None
    for i, path in enumerate(gen):
        paths = path
        if i == 1:
            break
    trans = parse_paths_ring(paths, all_react_ids)
    return trans


def return_chain_trans(rxn, all_react_ids):
    react, prod = rxn_to_graphs(rxn)
    gen = optimize_edit_paths(react, prod, node_match=node_match)
    paths = None
    for i, path in enumerate(gen):
        paths = path
        if i == 5:
            break
    trans = parse_paths_chain(paths, all_react_ids)
    return trans


def return_PO_trans(rxn):
    react, prod = rxn_to_graphs(rxn)
    gen = optimize_edit_paths(react, prod, node_match=node_match)
    paths = None
    for i, path in enumerate(gen):
        paths = path
        if i == 5:
            break
    print(paths)
    trans = parse_paths_PO(paths)
    return trans


def parse_paths_ring(paths, react_ids):
    trans = {"Ring_Breakage": 0, "Nb_Paths": 0}
    paths = [paths]
    for path in paths:
        is_break = False
        nodes, edges = path[0], path[1]
        trans["Nb_Paths"] += 1
        for edit_edge in edges:
            if is_ring_edge(edit_edge[0], react_ids) and ((edit_edge[1] is None) or (edit_edge[0][0][2] != edit_edge[1][0][2]) or (edit_edge[0][1][2] != edit_edge[1][1][2])) and not is_break:
                trans["Ring_Breakage"] += 1
                is_break = True
    return trans


def parse_paths_chain(paths, react_ids):
    paths = [paths]
    trans = {"Chain_Breakage": 0, "Nb_Paths": 0}
    for path in paths:
        is_break = False
        nodes, edges = path[0], path[1]
        trans["Nb_Paths"] += 1
        for edit_edge in edges:
            if is_chain_edge(edit_edge[0], react_ids) and ((edit_edge[1] is None) or (edit_edge[0][0][2] != edit_edge[1][0][2]) or (edit_edge[0][1][2] != edit_edge[1][1][2])) and not is_break:
                trans["Chain_Breakage"] += 1
                is_break = True
    return trans


def parse_paths_PO(paths):
    trans = {"PO_Breakage": 0, "PO_Bonding": 0, "Other": 0, "Nb_Paths": 0}
    paths = [paths]
    for path in paths:
        nodes, edges = path[0], path[1]
        trans["Nb_Paths"] += 1
        is_PO_break, is_PO_bond = False, False
        for edit_edge in edges:
            if is_PO_edge(edit_edge[0]) and ((edit_edge[1] is None) or not is_PO_edge(edit_edge[1])) and not is_PO_break:
                trans["PO_Breakage"] += 1
                is_PO_break = True
            elif is_PO_edge(edit_edge[1]) and ((edit_edge[0] is None) or not is_PO_edge(edit_edge[0])) and not is_PO_bond:
                trans["PO_Bonding"] += 1
                is_PO_bond = True
            else:
                trans["Other"] += 1
    return trans
