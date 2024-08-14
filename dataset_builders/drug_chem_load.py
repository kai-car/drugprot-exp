# dataset_builders/drug_chem_load.py

from typing import Dict, List, Union

import datasets
from pie_datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from pie_datasets.core.dataset import get_pie_dataset_type
from pytorch_ie.documents import (
    Document,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)


def _add_dset_name_to_document(doc: Document, name: str) -> Document:
    if not hasattr(doc, "metadata"):
        raise ValueError(
            f"Document does not have metadata attribute which required to save the dataset name: {doc}"
        )
    if "dataset_name" in doc.metadata:
        raise ValueError(
            f"Document already has a dataset_name attribute: {doc.metadata['dataset_name']}"
        )
    doc.metadata["dataset_name"] = name
    return doc


def concatenate_datasets(
    dsets: Union[
        List[Dataset], List[IterableDataset], Dict[str, Dataset], Dict[str, IterableDataset]
    ]
) -> Union[Dataset, IterableDataset]:
    """Concatenate multiple datasets into a single dataset.

    The datasets must have the same
    document type.
    Args:
        dsets: A list of datasets or a dictionary with dataset names as keys and datasets as values. If a dictionary is
            provided, the dataset names will be added to the documents as metadata.
    Returns:
        A new dataset that is the concatenation of the input datasets.
    """

    if isinstance(dsets, dict):
        dsets = [
            dset.map(_add_dset_name_to_document, fn_kwargs={"name": name})
            for name, dset in dsets.items()
        ]

    if len(dsets) == 0:
        raise ValueError("No datasets to concatenate")

    document_type = dsets[0].document_type
    for doc in dsets[1:]:
        if not doc.document_type == document_type:
            raise ValueError("All datasets must have the same document type to concatenate")

    result_hf = datasets.concatenate_datasets(dsets)
    pie_dataset_type = get_pie_dataset_type(dsets[0])

    return pie_dataset_type.from_hf_dataset(result_hf, document_type=document_type)


def load_drug_chem_dataset() -> DatasetDict:
    # Implement your custom dataset loading logic here
    # For example, reading CSV files, processing data, etc.

    drugprot = load_dataset("pie/drugprot", name="drugprot_source")
    chemprot = load_dataset("pie/chemprot", name="chemprot_full_source")

    drugprot = drugprot.to_document_type(
        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
    )
    chemprot = chemprot.to_document_type(
        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
    )

    train = concatenate_datasets([drugprot["train"], chemprot["train"]])
    validation = concatenate_datasets([drugprot["validation"], chemprot["validation"]])
    # test = concatenate_datasets([drugprot["test_background"], chemprot["test"]])

    return DatasetDict(
        {
            "train": train,
            # 'test': test,
            "validation": validation,
        }
    )
