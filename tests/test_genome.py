# pylint: disable=missing-docstring
from pathlib import Path

import pytest
from genome_spot.bioinformatics.genome import Genome


cwd = Path(__file__).resolve().parent

CONTIG_FASTA = f"{cwd}/test_data/GCA_000172155.1_ASM17215v1_genomic.fna.gz"
PROTEIN_FASTA = f"{cwd}/test_data/GCA_000172155.1_ASM17215v1_protein.faa.gz"


class TestGenome:

    def test_genome_init(self):
        with pytest.raises(FileNotFoundError):
            genome_calc = Genome(contig_filepath="fake.fna", protein_filepath=PROTEIN_FASTA)

        with pytest.raises(FileNotFoundError):
            genome_calc = Genome(contig_filepath=CONTIG_FASTA, protein_filepath="fake.faa")

    def test_measure_genome_features(self):

        genome_calc = Genome(
            contig_filepath=CONTIG_FASTA,
            protein_filepath=PROTEIN_FASTA,
        )

        genome_features = genome_calc.measure_genome_features()

        expected_keys = [
            "all",
            "extracellular_soluble",
            "intracellular_soluble",
            "membrane",
            "diff_extra_intra",
        ]

        expected_features = [
            "nt_length",
            "pur_pyr_transition_freq",
            "nt_C",
            "nt_A",
            "total_proteins",
            "total_protein_length",
            "pis_acidic",
            "pis_neutral",
            "pis_basic",
            "pis_3_4",
            "pis_4_5",
            "pis_5_6",
            "pis_6_7",
            "pis_7_8",
            "pis_8_9",
            "pis_9_10",
            "pis_10_11",
            "pis_11_12",
            "mean_pi",
            "mean_gravy",
            "mean_zc",
            "mean_nh2o",
            "mean_protein_length",
            "mean_thermostable_freq",
            "proportion_R_RK",
            "aa_M",
            "aa_S",
            "aa_L",
            "aa_Y",
            "aa_T",
            "aa_P",
            "aa_K",
            "aa_V",
            "aa_D",
            "aa_I",
            "aa_C",
            "aa_G",
            "aa_A",
            "aa_H",
            "aa_E",
            "aa_R",
            "aa_Q",
            "aa_N",
            "aa_F",
            "aa_W",
            "protein_coding_density",
        ]

        expected_values_all = {
            "nt_length": 8220859,
            "pur_pyr_transition_freq": 0.4796972480724635,
            "nt_C": 0.6031152226586596,
            "nt_A": 0.3968847773413404,
            "total_proteins": 6519,
            "total_protein_length": 2366909,
            "pis_acidic": 0.19880349746893694,
            "pis_neutral": 0.49148642429820527,
            "pis_basic": 0.3097100782328578,
            "pis_3_4": 0.0,
            "pis_4_5": 0.09633379352661452,
            "pis_5_6": 0.24144807485810707,
            "pis_6_7": 0.2202791839239147,
            "pis_7_8": 0.08712992790305261,
            "pis_8_9": 0.11228716060745512,
            "pis_9_10": 0.1774812087743519,
            "pis_10_11": 0.04509894155545329,
            "pis_11_12": 0.019941708851050774,
            "mean_pi": 7.200740246276107,
            "mean_gravy": -0.17479389716023352,
            "mean_zc": -0.3247701124075573,
            "mean_nh2o": 0.0062494883704452345,
            "mean_protein_length": 363.07853965332106,
            "mean_thermostable_freq": 0.3810484475744526,
            "proportion_R_RK": 0.5884605005465586,
            "aa_M": 0.02434063208440606,
            "aa_S": 0.06170794220213469,
            "aa_L": 0.10175385055529326,
            "aa_Y": 0.024041867119667687,
            "aa_T": 0.05536256720374455,
            "aa_P": 0.057010369588417334,
            "aa_K": 0.04550563241779971,
            "aa_V": 0.0726010380701561,
            "aa_D": 0.05028363621605396,
            "aa_I": 0.04365774953610717,
            "aa_C": 0.011265560139453169,
            "aa_G": 0.07959808376916672,
            "aa_A": 0.1020650426648756,
            "aa_H": 0.024495392414772013,
            "aa_E": 0.06152117209775906,
            "aa_R": 0.06506852262256695,
            "aa_Q": 0.036957344407316496,
            "aa_N": 0.028806341996696058,
            "aa_F": 0.03772186542835214,
            "aa_W": 0.01658094982206415,
            "protein_coding_density": 0.8637451390420393,
        }

        # check keys are correct
        assert list(genome_features.keys()) == expected_keys
        assert list(genome_features["all"].keys()) == expected_features

        # check proteins are found for all localizations
        for genome_statistics in genome_features.values():
            assert abs(genome_statistics["total_proteins"]) > 400

        # check values are calculated consistently
        assert genome_features["all"] == pytest.approx(expected_values_all)
