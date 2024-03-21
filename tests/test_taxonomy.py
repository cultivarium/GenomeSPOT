# pylint: disable=missing-docstring
from pathlib import Path

from genome_spot.taxonomy.taxonomy import TaxonomyGTDB


cwd = Path(__file__).resolve().parent
TAXONOMY_FILENAMES = [f"{cwd}/test_data/test_ar53_taxonomy_r214.tsv", f"{cwd}/test_data/test_bac120_taxonomy_r214.tsv"]


class TestTaxonomyGTDB:

    def test_convert_gtdb_to_ncbi(self):
        taxonomy = TaxonomyGTDB(TAXONOMY_FILENAMES)
        gtdb_accession = "RS_GCF_015350875.1"
        ncbi_accession = taxonomy.convert_gtdb_to_ncbi(gtdb_accession, make_genbank=True, remove_version=True)
        assert ncbi_accession == "GCA_015350875"

    def test_make_taxonomy_dict(self):
        taxonomy = TaxonomyGTDB(TAXONOMY_FILENAMES)
        assert len(taxonomy.taxonomy_dict) == 1100
        assert taxonomy.taxonomy_dict["GCA_000875775"] == (
            "Archaea",
            "Thermoproteota",
            "Nitrososphaeria",
            "Nitrososphaerales",
            "Nitrosopumilaceae",
            "Nitrosopumilus",
            "Nitrosopumilus piranensis",
        )
        assert taxonomy.taxonomy_dict["GCA_902645985"] == (
            "Bacteria",
            "Pseudomonadota",
            "Alphaproteobacteria",
            "Rickettsiales",
            "Anaplasmataceae",
            "Wolbachia",
            "Wolbachia pipientis",
        )

    def test_taxa_of_genomes(self):
        taxonomy = TaxonomyGTDB(TAXONOMY_FILENAMES)
        genomes = ["GCA_014729675", "GCA_006954425", "GCA_000875775"]
        taxa = taxonomy.taxa_of_genomes(genomes, "family")
        assert taxa == ["Methanomethylophilaceae", "Nitrosopumilaceae", "WJKL01"]

    def test_genomes_in_taxa(self):

        taxonomy = TaxonomyGTDB(TAXONOMY_FILENAMES)
        genomes = ["GCA_014729675", "GCA_006954425", "GCA_000875775"]
        taxa = taxonomy.taxa_of_genomes(genomes, "family")
        related_genomes = taxonomy.genomes_in_taxa(taxa, "family")

        expected_values = [
            "GCA_000437835",
            "GCA_000875775",
            "GCA_002495325",
            "GCA_002506665",
            "GCA_005787845",
            "GCA_006954425",
            "GCA_014384145",
            "GCA_014729675",
            "GCA_017461225",
            "GCA_021268345",
            "GCA_021634835",
            "GCA_902552015",
            "GCA_939801045",
        ]
        assert related_genomes == expected_values

    def test_measure_diversity(self):
        taxonomy = TaxonomyGTDB(TAXONOMY_FILENAMES)
        genomes = ["GCA_014729675", "GCA_006954425", "GCA_000875775"]
        diversity = taxonomy.measure_diversity(query_rank="family", diversity_rank="species", subset_genomes=genomes)
        assert diversity == {"Methanomethylophilaceae": 1, "Nitrosopumilaceae": 1, "WJKL01": 1}
