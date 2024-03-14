from collections import Counter

import pytest
from genome_spot.bioinformatics.protein import Protein


PROTEIN_SEQUENCE = "".join(
    """MAAQDVKQQTPYRVIQLEWDAEKGERNEAVGNFDELVTHHPKSNSDAHLVDGKVVGGQAG
RTLGVVGGEIQEIEVSKAGKDYGLRPDQVLLKKDFMLEDSRLPSGPSSRSLDVPSPVAGV
VGTVNTSKGLVDVLDREGGDVILRVRHMSPLHVKAGDQVEYGQALGVQGKQATGAIHVHM
EVDSRYYQHYENYVGDLVSGRLSIDAERRDRGIEPRPFVDDGTIRIGGSSEMVQKVQQTL
NAEGYRGADNQPLQEDGVYRLSMQAAVINYQQAHGLSQTGDIDPATLQQIAPRTFPPELN
REDHNATPTYRNLQGAVPSQDPLHRQAEEDVRRLEQSLGRDYDDNSARLAASSAHLAKAN
GLTQIDHVVLSNQTAAVGKGENVFVVQGALDNPAHLMAHMKTSDAIAQPVEQSLSQLQTL
SETQRQQQAQQQSQQQDQQQLSAPQHRMV*BUZZBUZZBUZZBUZZBUZZBUZZBUZZBUZ
BUZZBUZZBUZZBUZZBUZZBUZZBUZZBUZBUZZBUZZBUZZBUZZBUZZBUZZBUZZBUZ
""".split(
        "\n"
    )
)


class TestProtein:

    def test_protein_formatting(self):
        protein = Protein(PROTEIN_SEQUENCE)
        assert protein.sequence[-20:] == "QQQSQQQDQQQLSAPQHRMV"

    def test_protein_length(self):
        protein = Protein(PROTEIN_SEQUENCE)
        assert protein.length == 449

    def test_protein_metrics(self):
        protein = Protein(PROTEIN_SEQUENCE)
        expected_values = {
            "pi": 5.382106971740723,
            "zc": 0.19821826280623608,
            "nh2o": 0.07137639198218272,
            "gravy": -0.6514476614699332,
            "thermostable_freq": 0.356347438752784,
            "length": 449,
            "is_exported": False,
            "aa_M": 0.0200445434298441,
            "aa_A": 0.08463251670378619,
            "aa_Q": 0.10690423162583519,
            "aa_D": 0.0757238307349666,
            "aa_V": 0.0935412026726058,
            "aa_K": 0.0334075723830735,
            "aa_T": 0.0400890868596882,
            "aa_P": 0.0467706013363029,
            "aa_Y": 0.026726057906458798,
            "aa_R": 0.062360801781737196,
            "aa_I": 0.031180400890868598,
            "aa_L": 0.08240534521158129,
            "aa_E": 0.05790645879732739,
            "aa_W": 0.0022271714922048997,
            "aa_G": 0.08685968819599109,
            "aa_N": 0.035634743875278395,
            "aa_F": 0.011135857461024499,
            "aa_H": 0.035634743875278395,
            "aa_S": 0.066815144766147,
        }

        assert protein.protein_metrics() == pytest.approx(expected_values)
