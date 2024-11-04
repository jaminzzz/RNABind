from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from Graphein.graphein.rna.constants import RNA_BASES
from graphein.utils.utils import onek_encoding_unk


def nucleotide_one_hot(
    n,
    d: Dict[str, Any],
    return_array: bool = True,
    allowable_set: Optional[List[str]] = None,
) -> Union[pd.Series, np.ndarray]:
    """Adds a one-hot encoding of nucleotide types as a node attribute.

    :param n: node name, this is unused and only included for compatibility
        with the other functions
    :type n: str
    :param d: Node data.
    :type d: Dict[str, Any]
    :param return_array: If True, returns a numpy array of one-hot encoding,
        otherwise returns a pd.Series. Default is True.
    :type return_array: bool
    :param allowable_set: Specifies vocabulary of nucleotides.
    :return: One-hot encoding of nucleotide types.
    :rtype: Union[pd.Series, np.ndarray]
    """

    if allowable_set is None:
        allowable_set = RNA_BASES
        
    features = onek_encoding_unk(d["residue_name"], allowable_set)

    if return_array:
        features = np.array(features).astype(int)
    else:
        features = pd.Series(features).astype(int)
        features.index = allowable_set

    d["nucleotide_one_hot"] = features
    return features