# pylint: disable=missing-docstring
from genome_spot.bioinformatics.signal_peptide import SignalPeptideHMM


PARTIAL_SEQUENCE = "MNKTLIAAAVAGIVLLASNAQAQTVPEGYQLQQVLMMSRHNLRAPLANNG"


class TestSignalPeptide:

    def test_predict_signal_peptide(self):
        signal_peptide_model = SignalPeptideHMM()
        is_exported, signal_end_index = signal_peptide_model.predict_signal_peptide(PARTIAL_SEQUENCE)
        signal_peptide = PARTIAL_SEQUENCE[: signal_end_index + 1]
        assert is_exported is True
        assert signal_peptide == "MNKTLIAAAVAGIVLLASNAQA"
