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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, insert_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
       "тысяча один" ->  cardinal { integer: "1 001" }

    Args:
        tn_cardinal: Text normalization Cardinal graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph = tn_cardinal.cardinal_numbers_default
        self.graph = graph.invert().optimize()

        repeat_count_words = {
            2: ["два", "две"],
            3: ["три"],
            4: ["четыре"],
            5: ["пять"],
            6: ["шесть"],
            7: ["семь"],
            8: ["восемь"],
            9: ["девять"],
        }
        repeated_digit_nouns = {
            "0": ["ноль", "нуля", "нулей"],
            "1": ["единица", "единицы", "единиц"],
            "2": ["двойка", "двойки", "двоек"],
            "3": ["тройка", "тройки", "троек"],
            "4": ["четверка", "четверки", "четверок"],
            "5": ["пятерка", "пятерки", "пятерок"],
            "6": ["шестерка", "шестерки", "шестерок"],
            "7": ["семерка", "семерки", "семерок"],
            "8": ["восьмерка", "восьмерки", "восьмерок"],
            "9": ["девятка", "девятки", "девяток"],
        }
        repeated_digits = pynini.string_map(
            [
                (f"{count_word} {digit_noun}", digit * repeat_count)
                for repeat_count, count_words in repeat_count_words.items()
                for count_word in count_words
                for digit, digit_nouns in repeated_digit_nouns.items()
                for digit_noun in digit_nouns
            ]
        )

        optional_sign = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("минус ", "\"-\"") + insert_space, 0, 1
        )

        # do not invert numbers less than 10
        # @chrnegor: Allow them too
        graph = pynini.compose(graph, NEMO_DIGIT ** (1, ...)) | repeated_digits
        graph = optional_sign + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
