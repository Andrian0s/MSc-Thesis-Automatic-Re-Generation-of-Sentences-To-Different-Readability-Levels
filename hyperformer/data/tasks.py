"""Implements different tasks and defines the processors to convert each dataset

to a sequence to sequence format."""
from collections import OrderedDict

import pandas as pd
import abc
import datasets
import functools
import logging
import numpy as np
import torch
from metrics import metrics
from typing import Callable, Dict, Mapping, List

from .utils import round_stsb_target, compute_task_max_decoding_length

logger = logging.getLogger(__name__)


class AbstractTaskDataset(abc.ABC):
    """Defines the abstract class for all the tasks.
    name: the name of the task.
    task_specific_config: specifies the special configuration needs
        to be passed to encoder when decoding each task. Since different
        tasks, have different output space, the maximum decoding length
        varies based on the tasks.
    preprocessor: a processor to convert the given dataset to the sequence
        to sequence format.
    metrics: specifies the metrics to evaluate the task based on them.
    split_to_data_split: since not all the time, different splits of the
        datasets are available, we define a mapping from the wanted split
        to the existing dataset splits.
    small_datasets_without_all_splits: List of strings, defines the name
        of all low-resource tasks in which not all train/test/validation
        splits are available.
    large_data_without_all_splits: List of strings, defines the name of
        all high-resource tasks in which not all train/test/validation
        splits are available.
    """
    name = NotImplemented
    task_specific_config: Dict = NotImplemented
    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}

    small_datasets_without_all_splits = ["cola", "wnli", "rte", "trec", "superglue-cb", "sick",
                                         "mrpc", "stsb", "imdb", "commonsense_qa", "superglue-boolq"]
    large_data_without_all_splits = ["yelp_polarity", "qqp", "qnli",
                                     "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "sst2"]

    def __init__(self, seed=42):
        self.seed = seed

    def get_sampled_split(self, split: int, n_obs: int = None):
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        split = self.split_to_data_split[split]
        dataset = self.load_dataset(split)
        total_size = len(dataset)
        n_obs = self.check_n_obs(n_obs, total_size)
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def get_shuffled_sampled_split(self, split: int, n_obs: int = None):
        # Defines the random generator.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(mapped_split)
        # shuffle the dataset and get the random samples.
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        dataset = self.select_dataset_samples(indices, dataset, n_obs=n_obs)
        return dataset

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def select_dataset_samples(self, indices, dataset, n_obs: int = None):
        """
        Given a dataset for the split, obtains the sample indices for this split
        and returns the subsampled dataset.
        :param indices: the selected indices.
        :param dataset: dataset corresponding to this split.
        :return: subsampled dataset.
        """
        n_obs = self.check_n_obs(n_obs, len(indices))
        indices = indices[:n_obs] if n_obs is not None else indices
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, split=split, script_version="1.2.0")

    def get_train_split_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["train"]
        dataset = self.load_dataset(mapped_split)
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        validation_size = 1000
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def get_half_validation_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["validation"]
        dataset = self.load_dataset(mapped_split)
        validation_size = len(dataset)
        indices = torch.randperm(validation_size, generator=generator).tolist()
        if split == "validation":
            return indices[:(validation_size // 2)]
        else:
            return indices[validation_size // 2:]

    def get_dataset(self, split, n_obs=None, readability_extra = None, readability_vector_style = None, add_prefix=True, split_validation_test=False, n_task_embedding_dim=None):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                # split = self.get_sampled_split(split, n_obs)
                if readability_extra is not None:
                    dataset = self.load_dataset(split=split, readability_extra=readability_extra)
                else:
                    dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        if readability_extra is not None:
            return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix, readability_vector_style=readability_vector_style, n_task_embedding_dim=n_task_embedding_dim),
                           remove_columns=dataset.column_names)
        else:
            return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names)

    def seq2seq_format(self, src_strs: List[str], tgt_strs: List[str],
                       add_prefix: bool = False, prefix: str = None):
        src_prefix = self.name if prefix is None else prefix
        src_strs = [src_prefix] + src_strs if add_prefix else src_strs
        return {"src_texts": ' '.join(src_strs),
                "tgt_texts": ' '.join(tgt_strs),
                "task": self.name}
    
    def make_readability_hypernetwork_input_vector(self, src_readability, tgt_readability, readability_vector_style, readability_map, n_task_embedding_dim):
        """
        Create a vector of the source and target readability scores, with a size of 32 for each pair of src_readability and tgt_readability.
        
        Parameters:
        src_readability (str): The readability level of the source text.
        tgt_readability (str): The readability level of the target text.
        readability_vector_style (str): Determines the format of the output readability vector ('both', 'source_only', 'target_only', or 'difference').
        readability_map (dict): A dictionary mapping readability levels to their corresponding scores.

        Returns:
        numpy.ndarray: A numpy array containing the calculated readability vector.
        """
        half_embedding_dim = int(n_task_embedding_dim / 2) # 32 for 64
        # Create the source and target readability vectors
        src_readability_vector = [1] * readability_map[src_readability] + [0] * (half_embedding_dim - readability_map[src_readability])
        tgt_readability_vector = [1] * readability_map[tgt_readability] + [0] * (half_embedding_dim  - readability_map[tgt_readability])

        if readability_vector_style == 'both' or readability_vector_style == 'separate': # This is just to hadnle this extra case, to be removed in the future.
            readability_vector_np = np.array(src_readability_vector + tgt_readability_vector)
        elif readability_vector_style == 'source_only':
            readability_vector_np = np.array(src_readability_vector + [0] * half_embedding_dim)
        elif readability_vector_style == 'target_only':
            readability_vector_np = np.array(tgt_readability_vector + [0] * half_embedding_dim)
        elif readability_vector_style == 'difference':  # readability_vector_style is difference of the two, as a 64-dimensional vector
            difference = readability_map[src_readability] - readability_map[tgt_readability]
            sign = -1 if difference > 0 else 1
            readability_vector = [sign] * abs(difference) + [0] * (n_task_embedding_dim - abs(difference))
            readability_vector_np = np.array(readability_vector)
        else: # We should never get here, but just in case
            raise ValueError('Invalid input_style: ' + readability_vector_style)
        return readability_vector_np
    
class OneStopParallelTextMappingClass(AbstractTaskDataset):
    name = 'onestop_parallel_text_plhsrc_plhtgt'
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    metrics = [
    metrics.LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE,
    metrics.SRC_PRED_EXACT_COPY_PERCENTAGE,
    metrics.AVG_PRED_TGT_LEV_DIST,
    metrics.AVG_GFI_SRC_PRED_DIFF,
    metrics.AVG_FRE_SRC_PRED_DIFF,
    metrics.AVG_FKGL_SRC_PRED_DIFF,
    metrics.AVG_ARI_SRC_PRED_DIFF,
    metrics.AVG_DCRF_SRC_PRED_DIFF,
    metrics.AVG_SMOG_SRC_PRED_DIFF,
    metrics.AVG_ASL_SRC_PRED_DIFF,
    metrics.AVG_GFI_PRED_TGT_ABSDIFF,
    metrics.AVG_FRE_PRED_TGT_ABSDIFF,
    metrics.AVG_FKGL_PRED_TGT_ABSDIFF,
    metrics.AVG_ARI_PRED_TGT_ABSDIFF,
    metrics.AVG_DCRF_PRED_TGT_ABSDIFF,
    metrics.AVG_SMOG_PRED_TGT_ABSDIFF,
    metrics.AVG_ASL_PRED_TGT_ABSDIFF,
    metrics.PRED_ONLY_GFI,
    metrics.PRED_ONLY_FRE,
    metrics.PRED_ONLY_FKGL,
    metrics.PRED_ONLY_ARI,
    metrics.PRED_ONLY_DCRF,
    metrics.PRED_ONLY_SMOG,
    metrics.PRED_ONLY_ASL,
    metrics.rouge,
    ]


    name_to_prefix = {
        'onestop_parallel_text_adv_int' : 'Simplify from advanced to intermediate: ',
        'onestop_parallel_text_adv_ele' : 'Simplify from advanced to elementary: ',
        'onestop_parallel_text_int_ele' : 'Simplify from intermediate to elementary: ',
        'onestop_parallel_text_ele_int' : 'Rewrite from elementary to intermediate: ',
        'onestop_parallel_text_ele_adv' : 'Rewrite from elementary to advanced: ',
        'onestop_parallel_text_int_adv' : 'Rewrite from intermediate to advanced: ',
    }

    def load_dataset(self, split, readability_extra):
        self.name = 'onestop_parallel_text_' + readability_extra
        # Read the filtered DataFrame into a Hugging Face Dataset object
        df = pd.read_csv(f"data/onestop_dataset/text_level/{readability_extra}/parallel_{split}.csv")
        hf_dataset = datasets.Dataset.from_pandas(df)
        return hf_dataset
    
    def seq2seq_format(self, src_strs: List[str], tgt_strs: List[str], readability_vector: np.ndarray,
                    add_prefix: bool = False, prefix: str = None):
        src_prefix = self.name if prefix is None else prefix
        src_strs = [src_prefix] + src_strs if add_prefix else src_strs
        return {"src_texts": ' '.join(src_strs),
                "tgt_texts": ' '.join(tgt_strs),
                "task": self.name,
                "readability_vector": readability_vector}

    def preprocessor(self, example, readability_vector_style, n_task_embedding_dim, add_prefix=False):
        # Extract the source and target texts from the provided example
        src_texts = [example['SourceText']]
        tgt_texts = [example['TargetText']]
        readability_map = {'ADV': int(n_task_embedding_dim  / 3), 'INT': int(n_task_embedding_dim / 4), 'ELE': int(n_task_embedding_dim / 8 )} # 64 is ADV 24 -> INT 16 , -> ELE 8 
        readability_vectors = self.make_readability_hypernetwork_input_vector(src_readability = example['SourceLevel'], tgt_readability = example['TargetLevel'], readability_vector_style = readability_vector_style, readability_map = readability_map, n_task_embedding_dim = n_task_embedding_dim)
        # Convert the source and target texts into the seq2seq format using the seq2seq_format() method
        # This method combines the texts and adds an optional prefix to the source text
        return self.seq2seq_format(src_texts, tgt_texts, readability_vectors, add_prefix, prefix=self.name_to_prefix[self.name])

class OneStopParallelSentenceMappingClass(AbstractTaskDataset):
    name = 'onestop_parallel_sentence_plhsrc_plhtgt'
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    metrics = [
    metrics.SARI,
    metrics.LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE,
    metrics.CORRECT_PARAPHRASE_PERCENTAGE,
    metrics.SRC_PRED_EXACT_COPY_PERCENTAGE,
    metrics.AVG_PRED_TGT_LEV_DIST,
    metrics.AVG_GFI_SRC_PRED_DIFF,
    metrics.AVG_FRE_SRC_PRED_DIFF,
    metrics.AVG_FKGL_SRC_PRED_DIFF,
    metrics.AVG_ARI_SRC_PRED_DIFF,
    metrics.AVG_DCRF_SRC_PRED_DIFF,
    metrics.AVG_SMOG_SRC_PRED_DIFF,
    metrics.AVG_ASL_SRC_PRED_DIFF,
    metrics.AVG_GFI_PRED_TGT_ABSDIFF,
    metrics.AVG_FRE_PRED_TGT_ABSDIFF,
    metrics.AVG_FKGL_PRED_TGT_ABSDIFF,
    metrics.AVG_ARI_PRED_TGT_ABSDIFF,
    metrics.AVG_DCRF_PRED_TGT_ABSDIFF,
    metrics.AVG_SMOG_PRED_TGT_ABSDIFF,
    metrics.AVG_ASL_PRED_TGT_ABSDIFF,
    metrics.PRED_ONLY_GFI,
    metrics.PRED_ONLY_FRE,
    metrics.PRED_ONLY_FKGL,
    metrics.PRED_ONLY_ARI,
    metrics.PRED_ONLY_DCRF,
    metrics.PRED_ONLY_SMOG,
    metrics.PRED_ONLY_ASL,
    metrics.rouge,
    ]

    name_to_prefix = {
        'onestop_parallel_sentence_adv_int' : 'Simplify from advanced to intermediate: ',
        'onestop_parallel_sentence_adv_ele' : 'Simplify from advanced to elementary: ',
        'onestop_parallel_sentence_int_ele' : 'Simplify from intermediate to elementary: ',
        'onestop_parallel_sentence_ele_int' : 'Rewrite from elementary to intermediate: ',
        'onestop_parallel_sentence_ele_adv' : 'Rewrite from elementary to advanced: ',
        'onestop_parallel_sentence_int_adv' : 'Rewrite from intermediate to advanced: ',
    }


    def load_dataset(self, split, readability_extra):
        self.name = 'onestop_parallel_sentence_' + readability_extra
        # Read the filtered DataFrame into a Hugging Face Dataset object
        df = pd.read_csv(f"data/onestop_dataset/sentence_level/{readability_extra}/parallel_{split}.csv")
        hf_dataset = datasets.Dataset.from_pandas(df)
        return hf_dataset
    
    def seq2seq_format(self, src_strs: List[str], tgt_strs: List[str], readability_vector: np.ndarray,
                    add_prefix: bool = False, prefix: str = None):
        src_prefix = self.name if prefix is None else prefix
        src_strs = [src_prefix] + src_strs if add_prefix else src_strs
        return {"src_texts": ' '.join(src_strs),
                "tgt_texts": ' '.join(tgt_strs),
                "task": self.name,
                "readability_vector": readability_vector}

    def preprocessor(self, example, readability_vector_style, n_task_embedding_dim = 64, add_prefix=False):
        # Extract the source and target texts from the provided example
        src_texts = [example['SourceText']]
        tgt_texts = [example['TargetText']]
        readability_map = {'ADV': int(n_task_embedding_dim  / 3), 'INT': int(n_task_embedding_dim / 4), 'ELE': int(n_task_embedding_dim / 8 )} # 64 is ADV 24 -> INT 16 , -> ELE 8 
        readability_vectors = self.make_readability_hypernetwork_input_vector(src_readability = example['SourceLevel'], tgt_readability = example['TargetLevel'], readability_vector_style = readability_vector_style, readability_map = readability_map, n_task_embedding_dim = n_task_embedding_dim)
        # Convert the source and target texts into the seq2seq format using the seq2seq_format() method
        # This method combines the texts and adds an optional prefix to the source text
        return self.seq2seq_format(src_texts, tgt_texts, readability_vectors, add_prefix, prefix=self.name_to_prefix[self.name])
    

class NewsElaParallelSentenceMappingClass(AbstractTaskDataset):
    name = 'newsela_parallel_sentence_plhsrc_plhtgt'
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    metrics = [
    metrics.SARI,
    metrics.LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE,
    metrics.CORRECT_PARAPHRASE_PERCENTAGE,
    metrics.SRC_PRED_EXACT_COPY_PERCENTAGE,
    metrics.AVG_PRED_TGT_LEV_DIST,
    metrics.AVG_GFI_SRC_PRED_DIFF,
    metrics.AVG_FRE_SRC_PRED_DIFF,
    metrics.AVG_FKGL_SRC_PRED_DIFF,
    metrics.AVG_ARI_SRC_PRED_DIFF,
    metrics.AVG_DCRF_SRC_PRED_DIFF,
    metrics.AVG_SMOG_SRC_PRED_DIFF,
    metrics.AVG_ASL_SRC_PRED_DIFF,
    metrics.AVG_GFI_PRED_TGT_ABSDIFF,
    metrics.AVG_FRE_PRED_TGT_ABSDIFF,
    metrics.AVG_FKGL_PRED_TGT_ABSDIFF,
    metrics.AVG_ARI_PRED_TGT_ABSDIFF,
    metrics.AVG_DCRF_PRED_TGT_ABSDIFF,
    metrics.AVG_SMOG_PRED_TGT_ABSDIFF,
    metrics.AVG_ASL_PRED_TGT_ABSDIFF,
    metrics.PRED_ONLY_GFI,
    metrics.PRED_ONLY_FRE,
    metrics.PRED_ONLY_FKGL,
    metrics.PRED_ONLY_ARI,
    metrics.PRED_ONLY_DCRF,
    metrics.PRED_ONLY_SMOG,
    metrics.PRED_ONLY_ASL,
    metrics.rouge,
    ]

    name_to_prefix = {
        'newsela_parallel_sentence_Level0_Level1' : 'Simplify from advanced to intermediate: ',
        'newsela_parallel_sentence_Level0_Level3' : 'Simplify from advanced to elementary: ',
        'newsela_parallel_sentence_Level1_Level3' : 'Simplify from intermediate to elementary: ',
        'newsela_parallel_sentence_Level3_Level1' : 'Rewrite from elementary to intermediate: ',
        'newsela_parallel_sentence_Level3_Level0' : 'Rewrite from elementary to advanced: ',
        'newsela_parallel_sentence_Level1_Level0' : 'Rewrite from intermediate to advanced: ',
    }

    def load_dataset(self, split, readability_extra):
        self.name = 'newsela_parallel_sentence_' + readability_extra
        # Read the filtered DataFrame into a Hugging Face Dataset object
        df = pd.read_csv(f"data/newsela_dataset/sentence_level/{readability_extra}/parallel_{split}.csv")
        hf_dataset = datasets.Dataset.from_pandas(df)
        return hf_dataset
    
    def seq2seq_format(self, src_strs: List[str], tgt_strs: List[str], readability_vector: np.ndarray,
                    add_prefix: bool = False, prefix: str = None):
        src_prefix = self.name if prefix is None else prefix
        src_strs = [src_prefix] + src_strs if add_prefix else src_strs
        return {"src_texts": ' '.join(src_strs),
                "tgt_texts": ' '.join(tgt_strs),
                "task": self.name,
                "readability_vector": readability_vector}

    def preprocessor(self, example, readability_vector_style, n_task_embedding_dim = 64, add_prefix=False):
        # Extract the source and target texts from the provided example
        src_texts = [example['SourceText']]
        tgt_texts = [example['TargetText']]
        readability_map = {'Level0': int(n_task_embedding_dim  / 3), 'Level1': int(n_task_embedding_dim / 4), 'Level3': int(n_task_embedding_dim / 8 )} # 64 is Level0 24 -> Level1 16 , -> Level0 8 
        readability_vectors = self.make_readability_hypernetwork_input_vector(src_readability = example['SourceLevel'], tgt_readability = example['TargetLevel'], readability_vector_style = readability_vector_style, readability_map = readability_map, n_task_embedding_dim = n_task_embedding_dim)
        # Convert the source and target texts into the seq2seq format using the seq2seq_format() method
        # This method combines the texts and adds an optional prefix to the source text
        return self.seq2seq_format(src_texts, tgt_texts, readability_vectors, add_prefix, prefix=self.name_to_prefix[self.name])
    
class IMDBTaskDataset(AbstractTaskDataset):
    name = "imdb"
    split_to_data_split = {"train": "train",
                           "validation": "test",
                           "test": "test"}
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["text"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
   
class SickTaskDataset(AbstractTaskDataset):
    name = "sick"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    label_to_target = {"ENTAILMENT": 0, "CONTRADICTION": 2, "NEUTRAL": 1}
    metrics = [metrics.accuracy]

    def load_dataset(self, split: int):
        return datasets.load_dataset("csv", data_files={split: f"sick/{split}_clean.csv"})[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(self.label_to_target[example["label"]])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class PawsTaskDataset(AbstractTaskDataset):
    name = "paws"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, 'labeled_final', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"], "sentence2:", example["sentence2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEBoolQTaskDataset(AbstractTaskDataset):
    name = "superglue-boolq"
    label_list = ['0', '1']
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'boolq', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"], "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERTETaskDataset(AbstractTaskDataset):
    name = "superglue-rte"
    label_list = ['0', '1']
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'rte', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECBTaskDataset(AbstractTaskDataset):
    name = "superglue-cb"
    label_list = ['0', '1', '2']
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'cb', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SNLITaskDataset(AbstractTaskDataset):
    name = "snli"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class IWSLT2017RONL(AbstractTaskDataset):
    name = "iwslt2017-ro-nl"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-nl"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-ro-nl',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["ro"]]
        tgt_texts = [example['translation']["nl"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate Romanian to Dutch")


class IWSLT2017ENNL(AbstractTaskDataset):
    name = "iwslt2017-en-nl"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"en-nl"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-en-nl',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["nl"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Dutch")


class WMT16ENROTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-ro"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["ro"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Romanian")


class WMT16ROENTaskDataset(AbstractTaskDataset):
    name = "wmt16-ro-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["ro"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate Romanian to English")


class WMT16ENCSTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-cs"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"cs-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["cs"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Czech")


class WMT16ENFITaskDataset(AbstractTaskDataset):
    name = "wmt16-en-fi"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"fi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["fi"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Finnish")


class WMT14HIENTaskDataset(AbstractTaskDataset):
    name = "wmt14-hi-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"hi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["hi"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Hindi")


class TRECTaskDataset(AbstractTaskDataset):
    name = "trec"
    label_list = ["0", "1", "2", "3", "4", "5"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "test",
                           "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset("trec", split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label-coarse'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class YelpPolarityTaskDataset(AbstractTaskDataset):
    name = "yelp_polarity"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "test",
                           "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset("yelp_polarity",
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class ScitailTaskDataset(AbstractTaskDataset):
    name = "scitail"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    label_map = {"entailment": 0, "neutral": 1}

    def map_label(self, label):
        return self.label_map[label]

    def load_dataset(self, split):
        return datasets.load_dataset("scitail", "snli_format",
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'], "sentence2:", example["sentence2"]]
        # To increase the transfer performance, we modified the targets to be similar to other datasets.
        tgt_texts = [str(self.map_label(example['gold_label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MRPCTaskDataset(AbstractTaskDataset):
    name = "mrpc"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc',
                                     split=split, script_version="1.2.0")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLATaskDataset(AbstractTaskDataset):
    name = "cola"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.matthews_corrcoef]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2TaskDataset(AbstractTaskDataset):
    name = "sst2"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSBTaskDataset(AbstractTaskDataset):
    name = "stsb"
    label_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQPTaskDataset(AbstractTaskDataset):
    name = "qqp"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLITaskDataset(AbstractTaskDataset):
    name = "mnli"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLITaskDataset(AbstractTaskDataset):
    name = "qnli"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class RTETaskDataset(AbstractTaskDataset):
    name = "rte"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLITaskDataset(AbstractTaskDataset):
    name = "wnli"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SocialIQaTaskDataset(AbstractTaskDataset):
    name = "social_i_qa"
    label_map = {"1": "0", "2": "1", "3": "2"}
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "context:", example["context"],
                     "answerA:", example["answerA"],
                     "answerB:", example["answerB"],
                     "answerC:", example["answerC"]]
        tgt_texts = [self.label_map[example['label'].rstrip()]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class CosmosQaTaskDataset(AbstractTaskDataset):
    name = "cosmos_qa"
    label_list = ["0", "1", "2", "3"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "context:", example["context"],
                     "answer0:", example["answer0"],
                     "answer1:", example["answer1"],
                     "answer2:", example["answer2"],
                     "answer3:", example["answer3"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WinograndeTaskDataset(AbstractTaskDataset):
    name = "winogrande"
    label_list = ["1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('winogrande', 'winogrande_l',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["sentence"],
                     "option1:", example["option1"],
                     "option2:", example["option2"]]
        tgt_texts = [str(example['answer'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class HellaSwagTaskDataset(AbstractTaskDataset):
    name = "hellaswag"
    label_list = ["0", "1", "2", "3"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["ctx:", example["ctx"],
                     "ending0:", example["endings"][0],
                     "ending1:", example["endings"][1],
                     "ending2:", example["endings"][2],
                     "ending3:", example["endings"][3]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class CommonsenseQaTaskDataset(AbstractTaskDataset):
    name = "commonsense_qa"
    label_map = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}
    label_list = ["0", "1", "2", "3", "4"]  # ["A", "B", "C", "D", "E"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "A:", example["choices"]["text"][0],
                     "B:", example["choices"]["text"][1],
                     "C:", example["choices"]["text"][2],
                     "D:", example["choices"]["text"][3],
                     "E:", example["choices"]["text"][4]]
        tgt_texts = [str(self.label_map[example['answerKey']])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


TASK_MAPPING = OrderedDict([
    ('onestop_parallel_sentence_', OneStopParallelSentenceMappingClass) ,
    ('newsela_parallel_sentence_', NewsElaParallelSentenceMappingClass) ,
    ('onestop_parallel_text_', OneStopParallelTextMappingClass),
    ('superglue-boolq', SuperGLUEBoolQTaskDataset),
    ('superglue-cb', SuperGLUECBTaskDataset),
    ('superglue-rte', SuperGLUERTETaskDataset),
    ('paws', PawsTaskDataset),
    ('imdb', IMDBTaskDataset),
    ('snli', SNLITaskDataset),
    ('scitail', ScitailTaskDataset),
    ('mrpc', MRPCTaskDataset),
    ('trec', TRECTaskDataset),
    ('yelp_polarity', YelpPolarityTaskDataset),
    ('wmt16-ro-en', WMT16ROENTaskDataset),
    ('wmt14-hi-en', WMT14HIENTaskDataset),
    ('wmt16-en-ro', WMT16ENROTaskDataset),
    ('wmt16-ro-en', WMT16ROENTaskDataset),
    ('wmt16-en-cs', WMT16ENCSTaskDataset),
    ('iwslt2017-ro-nl', IWSLT2017RONL),
    ('iwslt2017-en-nl', IWSLT2017ENNL),
    ('cola', COLATaskDataset),
    ('sst2', SST2TaskDataset),
    ('stsb', STSBTaskDataset),
    ('qqp', QQPTaskDataset),
    ('mnli', MNLITaskDataset),
    ('qnli', QNLITaskDataset),
    ('rte', RTETaskDataset),
    ('wnli', WNLITaskDataset),
    ('wmt16-en-fi', WMT16ENFITaskDataset),
    ('social_i_qa', SocialIQaTaskDataset),
    ('cosmos_qa', CosmosQaTaskDataset),
    ('winogrande', WinograndeTaskDataset),
    ('hellaswag', HellaSwagTaskDataset),
    ('commonsense_qa', CommonsenseQaTaskDataset),
    ('sick', SickTaskDataset)]
)


class AutoTask:
    @classmethod
    def get(self, task_name, seed=42):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
