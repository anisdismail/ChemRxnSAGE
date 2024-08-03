import pandas as pd
import numpy as np

from preprocessing import create_fingerprints, decode_ids_np
from utils import load_rxn_classifier, predict_rxn_type
from metrics import filter_0, filter_2, filter_4, filter_5, similarity, is_valid_rxn, JSD_with_train


"""
Filters Pipeline
"""
pd.options.mode.use_inf_as_na = True


def validity_statistics(df):
    val = df["isValid"].mean()
    fil0 = df["Filter_0"].mean()
    fil2 = df["Filter_2"].mean()
    fil4 = df["Filter_4"].mean()
    fil5 = df["Filter_5"].mean()
    return val, fil0, fil2, fil4, fil5


def filters_pipeline(df):
    # run the molecular validation first
    df["isValid"] = df["decoded_smiles"].apply(is_valid_rxn)
    # run the heuristics filters
    df["Filter_0"] = df.apply(lambda x: filter_0(
        x["decoded_smiles"]) if x["isValid"] else False, axis=1)
    df["Filter_2"] = df.apply(lambda x: filter_2(
        x["decoded_smiles"]) if x["isValid"] else False, axis=1)
    df["Filter_4"] = df.apply(lambda x: filter_4(
        x["decoded_smiles"]) if x["isValid"] else False, axis=1)
    df["Filter_5"] = df.apply(lambda x: filter_5(
        x["decoded_smiles"]) if x["isValid"] else False, axis=1)
    return df


def process_results(data, tokenizer, config, use_filters=True):
    df = pd.DataFrame()
    # decode ids into smiles
    df["decoded_smiles"] = np.apply_along_axis(decode_ids_np, 1, data, tokenizer)
    # run the decoded smiles into the filters pipeline
    df = filters_pipeline(df)
    # transform the data into fingerprints
    df, fingerprints, strings_decoded = create_fingerprints(
        df, tokenizer, use_filters, config)
    return df, fingerprints, strings_decoded


def generate_metrics_evaluation(generated_file, centroids, centroids_strings, tokenizer, config):
    with open(generated_file, "r", encoding="utf-8") as f:
        gen_data = np.loadtxt(f)
    df, gen_fingerprints, gen_strings = process_results(
        gen_data, tokenizer, config, )
    rxn_classifier = load_rxn_classifier(config)
    rxn_pred = predict_rxn_type(rxn_classifier, gen_fingerprints)
    avg_similarity, sims = similarity(
        gen_fingerprints, centroids, metric="jaccard")
    avg_str_similarity, sims_str = similarity(
        gen_strings, centroids_strings, metric="cosine")
    jsd = JSD_with_train(rxn_pred)
    valid, filter0, filter2, filter4, filter5 = validity_statistics(df)

    return (jsd, avg_similarity, avg_str_similarity, valid,
            filter0, filter2, filter4, filter5,
            df, rxn_pred, sims, gen_fingerprints)


def generate_metrics_comparison(generated_file, centroids, centroids_strings, tokenizer, config):
    with open(generated_file, "r", encoding="utf-8") as f:
        gen_data = np.loadtxt(f)

    def process_and_print_results(use_filters):
        df, gen_fingerprints, gen_strings = process_results(
            gen_data, tokenizer, config, use_filters=use_filters)
        rxn_classifier = load_rxn_classifier(config)
        rxn_pred = predict_rxn_type(rxn_classifier, gen_fingerprints)
        avg_similarity, _ = similarity(
            gen_fingerprints, centroids, metric="jaccard")
        avg_str_similarity, _ = similarity(
            gen_strings, centroids_strings, metric="cosine")
        jsd = JSD_with_train(rxn_pred)
        val, filter0, filter2, filter4, filter5 = validity_statistics(df)

        filter_status = "Using Filters" if use_filters else "No Filters"
        print(filter_status, jsd, avg_similarity, avg_str_similarity,
              val, filter0, filter2, filter4, filter5)

        return df, rxn_pred, gen_fingerprints

    df_old, rxn_pred_old, gen_fingerprints1 = process_and_print_results(
        use_filters=False)
    df, rxn_pred, gen_fingerprints2 = process_and_print_results(
        use_filters=True)

    return df_old, rxn_pred_old, df, rxn_pred, gen_fingerprints1, gen_fingerprints2
