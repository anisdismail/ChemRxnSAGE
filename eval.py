import ast
import os

import numpy as np
import pandas as pd
import sentencepiece as spm

from metrics import (
    JSS_with_train,
    calculate_dataset_diversity,
    calculate_diversity_per_class,
    filter_1,
    filter_2,
    filter_3,
    filter_4,
    is_valid_rxn,
    novelty_percentage,
    similarity,
    unique_percentage,
)
from preprocessing import create_fingerprints
from utils import canoncalize_valid_rxns, load_rxn_classifier, predict_rxn_type

pd.options.mode.use_inf_as_na = True


class Evaluator:
    def __init__(self, config):
        self.config = config
        with open(
            os.path.join(config["main_dir"], "train", "centroids_200.data"),
            "r",
            encoding="utf-8",
        ) as f:
            self.centroids = np.loadtxt(f)
        with open(
            os.path.join(config["main_dir"], "train", "centroids_strings_200.data"),
            "r",
            encoding="utf-8",
        ) as f:
            self.centroids_strings = np.loadtxt(f)
        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m  --user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1"
        )
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load("m.model")
        self.seq_len = self.config["seq_len"]
        self.data_ref_path = self.config["data_ref_path"]
        self.main_path = self.config["main_dir"]
        self.df_ref = pd.read_csv(self.data_ref_path)
        self.df_ref.Canon_Reactants = self.df_ref.Canon_Reactants.apply(
            lambda x: ast.literal_eval(x)
        )
        self.df_ref.Canon_Products = self.df_ref.Canon_Products.apply(
            lambda x: ast.literal_eval(x)
        )
        # Convert the df_with_max_200 frozensets into a set for fast lookup
        self.df_ref["Reactant_Product_Frozenset"] = self.df_ref.apply(
            lambda row: (
                frozenset(row["Canon_Reactants"]),
                frozenset(row["Canon_Products"]),
            ),
            axis=1,
        )

        self.rxn_classifier = load_rxn_classifier(self.main_path)

    def load_generated_dataset(self, generated_file):
        with open(generated_file, "r", encoding="utf-8") as f:
            self.gen_data = np.loadtxt(f)

    def filters_pipeline(self):
        self.df["isValid"] = self.df["decoded_smiles"].apply(is_valid_rxn)
        self.df["validated"] = self.df["isValid"]
        self.df["Filter_1"] = self.df.apply(
            lambda x: filter_1(x["decoded_smiles"]) if x["validated"] else False, axis=1
        )
        self.df["validated"] &= self.df["Filter_1"]
        self.df["Filter_2"] = self.df.apply(
            lambda x: filter_2(x["decoded_smiles"]) if x["validated"] else False, axis=1
        )
        self.df["validated"] &= self.df["Filter_2"]
        self.df["Filter_3"] = self.df.apply(
            lambda x: filter_3(x["decoded_smiles"]) if x["validated"] else False, axis=1
        )
        self.df["validated"] &= self.df["Filter_3"]
        self.df["Filter_4"] = self.df.apply(
            lambda x: filter_4(x["decoded_smiles"]) if x["validated"] else False, axis=1
        )
        self.df["validated"] &= self.df["Filter_4"]

    def process_results(self, use_filters=True):
        self.df = pd.DataFrame()
        # decode ids into smiles
        # self.df["decoded_smiles"] = np.apply_along_axis(
        #   decode_ids_np, 1, self.gen_data, self.tokenizer)
        self.df["input_ids"] = [self.gen_data[i, :] for i in range(len(self.gen_data))]
        self.df["decoded_smiles"] = self.df["input_ids"].apply(
            lambda x: self.tokenizer.decode_ids(x.astype(int).tolist())
        )
        self.df["decoded_smiles"] = (
            self.df["decoded_smiles"]
            .str.replace("[PAD]", "")
            .str.replace("[EOS]", "")
            .str.replace("[BOS]", "")
            .str.replace(" ", "")
            .str.replace("⁇", "")
        )
        # run the decoded smiles into the filters pipeline
        self.filters_pipeline()
        # transform the data into fingerprints
        self.df, self.gen_fingerprints, self.rxn_ids = create_fingerprints(
            self.df, self.tokenizer, use_filters, self.seq_len
        )
        if self.df is None:
            return None
        # canoncalize reactions
        self.valid_df = canoncalize_valid_rxns(self.df)

    """

    def filters_pipeline(self):
        # Timing is_valid_rxn application
        start_time = time.time()
        self.df["isValid"] = self.df["decoded_smiles"].apply(is_valid_rxn)
        is_valid_time = time.time() - start_time
        print(f"is_valid_rxn took {is_valid_time:.4f} seconds")

        # Timing Filter_1 application
        start_time = time.time()
        self.df["Filter_1"] = self.df.apply(lambda x: filter_1(
            x["decoded_smiles"]) if x["isValid"] else False, axis=1)
        filter_1_time = time.time() - start_time
        print(f"filter_1 took {filter_1_time:.4f} seconds")

        # Timing filter_2 application
        start_time = time.time()
        self.df["Filter_2"] = self.df.apply(lambda x: filter_2(
            x["decoded_smiles"]) if x["isValid"] else False, axis=1)
        filter_2_time = time.time() - start_time
        print(f"filter_2 took {filter_2_time:.4f} seconds")

        # Timing Filter_3 application
        start_time = time.time()
        self.df["Filter_3"] = self.df.apply(lambda x: filter_3(
            x["decoded_smiles"]) if x["isValid"] else False, axis=1)
        filter_3_time = time.time() - start_time
        print(f"filter_3 took {filter_3_time:.4f} seconds")

        # Timing Filter_4 application
        start_time = time.time()
        self.df["Filter_4"] = self.df.apply(lambda x: filter_4(
            x["decoded_smiles"]) if x["isValid"] else False, axis=1)
        filter_4_time = time.time() - start_time
        print(f"filter_4 took {filter_4_time:.4f} seconds")

    def process_results(self, use_filters=True):
        self.df = pd.DataFrame()

        # Timing decode_ids_np
        start_time = time.time()
        self.df["decoded_smiles"] = np.apply_along_axis(
            decode_ids_np, 1, self.gen_data, self.tokenizer)
        # self.df["decoded_smiles"] = self.gen_data.apply(
        #    lambda row: decode_ids_np(row, self.tokenizer), axis=1)

        decode_time = time.time() - start_time
        print(f"Decoding IDs to SMILES took {decode_time:.4f} seconds")

        # Timing filters_pipeline
        start_time = time.time()
        self.filters_pipeline()
        filters_time = time.time() - start_time
        print(f"Filters pipeline took {filters_time:.4f} seconds")

        # Timing create_fingerprints
        start_time = time.time()
        self.df, self.gen_fingerprints, self.rxn_ids = create_fingerprints(
            self.df, self.tokenizer, use_filters, self.seq_len)
        create_fingerprints_time = time.time() - start_time
        print(
            f"Creating fingerprints took {create_fingerprints_time:.4f} seconds")

        # Timing canoncalize_valid_rxns
        start_time = time.time()
        self.valid_df = canoncalize_valid_rxns(self.df)
        canoncalize_time = time.time() - start_time
        print(
            f"Canonicalizing valid reactions took {canoncalize_time:.4f} seconds")
    """

    def validity_statistics(self):
        val = self.df["isValid"].mean()
        fil1 = self.df["Filter_1"].sum() / self.df["isValid"].sum()
        fil2 = self.df["Filter_2"].sum() / self.df["Filter_1"].sum()
        fil3 = self.df["Filter_3"].sum() / self.df["Filter_2"].sum()
        fil4 = self.df["Filter_4"].sum() / self.df["Filter_3"].sum()
        validated = self.df["validated"].mean()
        return val, fil1, fil2, fil3, fil4, validated

    def generate_metrics_evaluation(self, generated_file):
        self.results = {
            "avg_similarity": 0.0,
            "avg_str_similarity": 0.0,
            "jss": 0.0,
            "valid": 0.0,
            "filter1": 0.0,
            "filter2": 0.0,
            "filter3": 0.0,
            "filter4": 0.0,
            "validated": 0.0,
            "novelty_perc": 0.0,
            "unique_perc": 0.0,
            "average_inter_dissimilarity": 0.0,
            "vendi_score_k": 0.0,
            "vendi_score_k_inf": 0.0,
            "vendi_score_k_small": 0.0,
            "avg_vs_score_per_class": 0.0,
        }

        self.load_generated_dataset(generated_file)
        self.process_results(use_filters=True)
        if self.df is not None:
            self.rxn_pred = predict_rxn_type(self.rxn_classifier, self.gen_fingerprints)
            self.results["avg_similarity"], _ = similarity(
                self.gen_fingerprints, self.centroids, metric="jaccard"
            )
            self.results["avg_str_similarity"], _ = similarity(
                self.rxn_ids, self.centroids_strings, metric="cosine"
            )
            self.results["jss"] = JSS_with_train(self.rxn_pred)
            (
                self.results["valid"],
                self.results["filter1"],
                self.results["filter2"],
                self.results["filter3"],
                self.results["filter4"],
                self.results["validated"],
            ) = self.validity_statistics()
            self.results["novelty_perc"] = novelty_percentage(
                self.valid_df, self.df_ref
            )
            self.results["unique_perc"] = unique_percentage(self.valid_df)
            (
                vendi_score_k,
                vendi_score_k_inf,
                vendi_score_k_small,
                self.results["average_inter_dissimilarity"],
            ) = calculate_dataset_diversity(self.gen_fingerprints)
            (
                self.results["vendi_score_k"],
                self.results["vendi_score_k_inf"],
                self.results["vendi_score_k_small"],
            ) = (
                vendi_score_k / len(self.gen_fingerprints),
                vendi_score_k_inf / len(self.gen_fingerprints),
                vendi_score_k_small / len(self.gen_fingerprints),
            )
            self.div_class_df = calculate_diversity_per_class(
                self.rxn_pred, self.gen_fingerprints
            )
            self.results["avg_vs_score_per_class"] = self.div_class_df["VS_norm"].mean()
        results_str = self.format_results()
        return results_str

    """

    def generate_metrics_evaluation(self, generated_file):
        self.results = {}

        # Timing load_generated_dataset
        start_time = time.time()
        self.load_generated_dataset(generated_file)
        load_time = time.time() - start_time
        print(f"load_generated_dataset took {load_time:.4f} seconds")

        # Timing process_results
        start_time = time.time()
        self.process_results(use_filters=True)
        process_results_time = time.time() - start_time
        print(f"process_results took {process_results_time:.4f} seconds")

        # Timing predict_rxn_type
        start_time = time.time()
        self.rxn_pred = predict_rxn_type(
            self.rxn_classifier, self.gen_fingerprints)
        predict_rxn_type_time = time.time() - start_time
        print(f"predict_rxn_type took {predict_rxn_type_time:.4f} seconds")

        # Timing similarity for avg_similarity
        start_time = time.time()
        self.results["avg_similarity"], _ = similarity(
            self.gen_fingerprints, self.centroids, metric="jaccard")
        avg_similarity_time = time.time() - start_time
        print(
            f"similarity (avg_similarity) took {avg_similarity_time:.4f} seconds")

        # Timing similarity for avg_str_similarity
        start_time = time.time()
        self.results["avg_str_similarity"], _ = similarity(
            self.rxn_ids, self.centroids_strings, metric="cosine")
        avg_str_similarity_time = time.time() - start_time
        print(
            f"similarity (avg_str_similarity) took {avg_str_similarity_time:.4f} seconds")

        # Timing JSS_with_train
        start_time = time.time()
        self.results["jss"] = JSS_with_train(self.rxn_pred)
        jss_time = time.time() - start_time
        print(f"JSS_with_train took {jss_time:.4f} seconds")

        # Timing validity_statistics
        start_time = time.time()
        self.results["valid"], self.results["filter1"], self.results["filter2"], self.results[
            "filter3"], self.results["filter4"], self.results["validated"] = self.validity_statistics()
        validity_statistics_time = time.time() - start_time
        print(
            f"validity_statistics took {validity_statistics_time:.4f} seconds")

        # Timing exact_matches_percentage
        start_time = time.time()
        self.results["exact_perc"] = exact_matches_percentage(
            self.valid_df, self.df_ref)
        exact_matches_percentage_time = time.time() - start_time
        print(
            f"exact_matches_percentage took {exact_matches_percentage_time:.4f} seconds")

        # Timing percentage_duplicates
        start_time = time.time()
        self.results["duplicates_perc"] = percentage_duplicates(self.valid_df)
        percentage_duplicates_time = time.time() - start_time
        print(
            f"percentage_duplicates took {percentage_duplicates_time:.4f} seconds")

        # Timing calculate_dataset_diversity
        start_time = time.time()
        vendi_score_k, vendi_score_k_inf, vendi_score_k_small, self.results["average_inter_dissimilarity"] = calculate_dataset_diversity(
            self.gen_fingerprints)
        calculate_dataset_diversity_time = time.time() - start_time
        print(
            f"calculate_dataset_diversity took {calculate_dataset_diversity_time:.4f} seconds")

        self.results["vendi_score_k"], self.results["vendi_score_k_inf"], self.results["vendi_score_k_small"] = vendi_score_k / \
            len(self.gen_fingerprints), vendi_score_k_inf / \
            len(self.gen_fingerprints), vendi_score_k_small / \
            len(self.gen_fingerprints)

        # Timing calculate_diversity_per_class
        start_time = time.time()
        self.div_class_df = calculate_diversity_per_class(
            self.rxn_pred, self.gen_fingerprints)
        calculate_diversity_per_class_time = time.time() - start_time
        print(
            f"calculate_diversity_per_class took {calculate_diversity_per_class_time:.4f} seconds")

        self.results["avg_vs_score_per_class"] = self.div_class_df["VS_norm"].mean()

        # Timing format_results
        start_time = time.time()
        results_str = self.format_results()
        format_results_time = time.time() - start_time
        print(f"format_results took {format_results_time:.4f} seconds")

        return results_str
"""

    def format_results(self):
        results_str = (
            f"JSS={self.results['jss']:.4f}, "
            f"Sim={self.results['avg_similarity']:.4f}, "
            f"StrSim={self.results['avg_str_similarity']:.4f}, "
            f"Val={self.results['valid']:.4f},\n"
            f"NoveltyPerc={self.results['novelty_perc']:.4f}, "
            f"UniquePerc={self.results['unique_perc']:.4f}, "
            f"IntDiv={self.results['average_inter_dissimilarity']:.4f}, "
            f"OverallVal={self.results['validated']:.4f},\n "
            f"VS={self.results['vendi_score_k']:.4f}, "
            f"VS(q=0.1)={self.results['vendi_score_k_small']:.4f}, "
            f"VD(q=inf)={self.results['vendi_score_k_inf']:.4f}, "
            f"AvgVSPerClass={self.results['avg_vs_score_per_class']:.4f}"
        )
        return results_str
