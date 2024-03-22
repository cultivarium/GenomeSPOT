"""Class to perform prediction of signal peptide presence in proteins"""

import os

import joblib
import numpy as np


TRAINED_MODEL = os.path.dirname(__file__) + "/hmm/hmm_signal_peptide.joblib"

SYMBOLS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
STATES = ["B", "C", "C1", "C2", "C3", "H", "M", "N", "N1", "N2", "N3"]
N_TERMINUS_LENGTH = 50
SIGNAL_PEPTIDE_END_STATE = "C1"
THRESHOLD_LOG_PROB = -134.0


class SignalPeptideHMM:
    """
    Class to predict signal peptides in protein sequences of bacteria and archaea.

    See notebooks/make_signal_peptide_hmm.ipynb to understand how the model was produced

    References:
    - Eddy, S. What is a hidden Markov model?. Nat Biotechnol 22, 1315–1316 (2004).
    - Pantelis G. Bagos et. al, Combined prediction of Tat and Sec signal
      peptides with hidden Markov models, Bioinformatics (2010)
    - Nielsen, H., & Krogh, A. . Prediction of signal peptides and signal
      anchors by a hidden Markov model. In Ismb (1998).
    - For comparing accuracy to SignalP versions: Almagro Armenteros et al.
      SignalP 5.0 improves signal peptide predictions using deep neural networks. Nat Biotechnol (2019)

    Args:
        model_file: filename of pretrained HMM file, default: model provided in package
    """

    def __init__(self, model_file: str = TRAINED_MODEL):
        self.model = joblib.load(model_file)
        self.symbols = SYMBOLS
        self.states = STATES
        self.threshold_log_prob = THRESHOLD_LOG_PROB
        self.signal_end_state = SIGNAL_PEPTIDE_END_STATE
        self.nterminus_length = N_TERMINUS_LENGTH
        self.symbol_to_idx = dict(zip(self.symbols, range(len(self.symbols))))
        self.state_to_index = dict(zip(self.states, range(len(self.states))))
        self.idx_to_state = dict(zip(range(len(self.states)), self.states))

    def _format_protein_sequence(self, protein_sequence) -> np.array:
        default_symbol = self.symbol_to_idx.get("G")  # hack: replace weird codes with 'neutral' glycine
        protein_nterminus = protein_sequence[0 : self.nterminus_length]
        arr_sequence = np.array([self.symbol_to_idx.get(aa, default_symbol) for aa in protein_nterminus]).reshape(-1, 1)
        return arr_sequence

    def _predict_hidden_states(self, formatted_sequence: str):
        """Function to fit sequence against HMM"""
        log_prob, pred_state_indices = self.model.decode(formatted_sequence)
        pred_states = [self.idx_to_state[idx] for idx in pred_state_indices]
        return pred_states, log_prob

    def predict_signal_peptide(self, protein_sequence) -> tuple:
        """Uses an HMM model to predict signal peptides in bacteria and archaea"""
        # Score sequence against model
        input_sequence = self._format_protein_sequence(protein_sequence)
        if len(input_sequence) < self.nterminus_length:  # too short
            is_exported = False
            signal_end_index = -1
        else:
            pred_states, log_prob = self._predict_hidden_states(input_sequence)
            # Assess probability of being signal peptide
            has_cut_site = self.signal_end_state in pred_states
            is_exported = (log_prob > self.threshold_log_prob) and has_cut_site
            if is_exported is True:
                signal_end_index = pred_states.index(self.signal_end_state)
            else:
                signal_end_index = -1

        return is_exported, signal_end_index
