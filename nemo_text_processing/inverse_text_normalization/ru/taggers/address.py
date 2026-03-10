# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_space,
)


class AddressFst(GraphFst):
    """
    Finite state transducer for classifying address fragments, specifically house numbers with fractions.
    E.g.
        "двадцать шесть дробь один" -> address { name: "26/1" }
        "26 дробь 1" -> address { name: "26/1" }

    Args:
        preprocessing: Preprocessing graph (not used in this FST but kept for interface consistency)
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="address", kind="classify", deterministic=deterministic)

        # Reuse TN cardinal graph and invert it to get spoken-number -> digit.
        cardinal_graph = pynini.invert(tn_cardinal.cardinal_numbers_default).optimize()

        # Accepts both converted words ("twenty") and existing digits ("20")
        number_graph = cardinal_graph | pynini.closure(NEMO_DIGIT, 1)

        # Rule for converting "slash"
        # delete_space ensures there are no spaces around the slash in the output
        slash = delete_space + pynini.cross("дробь", "/") + delete_space

        # Pattern: Number + "slash" + Number -> "N/Y"
        graph = number_graph + slash + number_graph

        graph = pynutil.insert("name: \"") + graph + pynutil.insert("\"")
        self.fst = self.add_tokens(graph).optimize()
