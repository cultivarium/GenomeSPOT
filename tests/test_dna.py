# pylint: disable=missing-docstring
import pytest
from genome_spot.bioinformatics.dna import DNA


class TestDNA:

    def test_dna(self):
        dna = DNA("ATCG")
        assert dna.sequence == "ATCG"

    def test_reverse_complement(self):
        dna = DNA("ATCG")
        assert dna.reverse_complement("ATCG") == "CGAT"
        assert dna.reverse_complement("ATCGNN") == "NNCGAT"

    def test_dna_length(self):
        dna = DNA("ATCG")
        assert dna.length == 4

    def test_nt_1mer_frequencies(self):
        dna = DNA("ACTAGCGACTAGC")
        expected_values = {"A": 0.46153846153846156, "C": 0.5384615384615384}
        assert dna.nt_1mer_frequencies() == pytest.approx(expected_values)

    def test_nt_2mer_frequencies(self):
        dna = DNA("ACTAGCGACTAGC")
        expected_values = {
            "AC": 0.16666666666666666,
            "AG": 0.3333333333333333,
            "TA": 0.16666666666666666,
            "GC": 0.16666666666666666,
            "CG": 0.08333333333333333,
            "GA": 0.08333333333333333,
        }
        assert dna.nt_2mer_frequencies() == pytest.approx(expected_values)

    def test_nucleotide_metrics(self):
        dna = DNA("ACTAGCGACTAGC")
        expected_values = {
            "nt_length": 13,
            "pur_pyr_transition_freq": 0.5833333333333334,
            "nt_A": 0.46153846153846156,
            "nt_C": 0.5384615384615384,
        }
        assert dna.nucleotide_metrics() == pytest.approx(expected_values)
