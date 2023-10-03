import sys
import numpy as np
from rdkit import Chem

from utils import convert_fingerprint


def create_fingerprints(df, tokenizer, use_filters, config):
    pad_token = tokenizer.encode_as_ids("[PAD]")[1]
    # only process filtered data
    df["validated"] = df["isValid"]
    if use_filters:
        df["validated"] = df["isValid"] & df["Filter_0"] & df["Filter_2"] & df["Filter_4"] & df["Filter_5"]
    validated = df["validated"].sum()
    if validated == 0:
        print("No Valid Reactions, exiting...")
        sys.exit(0)
    # encode the strings into ids
    df["decoded_smiles_delimited"] = "[BOS]"+df["decoded_smiles"]+"[EOS]"
    df["input_ids"] = df.apply(lambda x: tokenizer.encode_as_ids(x["decoded_smiles"])[
                               1:] if x["validated"] else "0", axis=1).fillna("0")

    # calculate how much to pad for each case
    df["input_counts"] = config["max_seq_len"] - \
        df["input_ids"].str.len().astype(int)

    # pad with the pad_token
    df["input_ids"] = df.apply(lambda x: np.pad(x["input_ids"], (0, x["input_counts"]),
                               mode='constant', constant_values=(0, pad_token)) if x["validated"] else 0, axis=1)

    params = Chem.rdChemReactions.ReactionFingerprintParams()
    params.fpSize = 2048
    df["rxn"] = df.apply(lambda x: Chem.rdChemReactions.ReactionFromSmarts(
        x["decoded_smiles"], useSmiles=True) if x["validated"] else 0, axis=1)
    df["fingerprint"] = df.apply(lambda x: Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(
        x["rxn"], params) if x["validated"] else 0, axis=1)

    # Filter the DataFrame to include only rows with "validated" True
    filtered_df = df[df["validated"]].copy()
    # Extract the "input_ids" column
    strings_decoded = filtered_df["input_ids"].values
    # Use the apply method to convert all fingerprints
    fingerprints = filtered_df.apply(convert_fingerprint, axis=1).values

    if len(fingerprints) > 0:
        return df, np.stack(fingerprints), np.stack(strings_decoded)
    return df, fingerprints, strings_decoded


def decode_ids_np(sequence, tokenizer):
    decoded = tokenizer.decode_ids(sequence.astype(int).tolist())
    for token in ["[PAD]", "[EOS]", "[BOS]", " ⁇ ", "⁇", " "]:
        decoded = decoded.replace(token, '')

    return decoded
