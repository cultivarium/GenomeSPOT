"""Queries BacDive API and computes trait values

Provides information for all strains up to the highest provided strain ID. Typical 
usage is run as a script (runtime: about 1 sec per 100 entries):

```shell
python3 src/query_bacdive.py -c .bacdive_credentials -max 171000 \
        -o bacdive_data.json
```
"""

import argparse
import json
import logging
import re
from typing import (
    Dict,
    Tuple,
)

import bacdive
import numpy as np
import pandas as pd


FREQUENT_OPTIMUM_PH = [6.0, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.5, 9.0]
FREQUENT_OPTIMUM_TEMP = [20.0, 25.0, 26.0, 26.5, 27.5, 28.0, 29.0, 30.0, 31.0, 32.5, 33.5, 35.0, 36.0, 37.0, 40.0]
FREQUENT_OPTIMUM_SALINITY = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.5, 10.0]


class QueryBacDive:
    """Downloads all strain data on the BacDive API.

    Queries all BacDive IDs from 0 to the highest ID available, which
    as of 2023-03 was about 171000. IDs appear to be sequential.

    Typical usage:
    ```
    bacdive_dict = QueryBacDive(
            username=username,
            password=password,
            max_bacdive_id=int(max_bacdive_id),
            min_bacdive_id=int(min_bacdive_id),
        ).scrape_bacdive_api()
    ```

    Args:
        username: BacDive username
        password: BacDive password
        min_bacdive_id: Minimum BacDive ID, default of 0 instead of 1
            for readability
        max_bacdive_id: Highest BacDive ID to query. Set above the highest
            available ID. See function `find_highest_bacdive_id`
    """

    def __init__(
        self,
        username: str,
        password: str,
        max_bacdive_id: int,
        min_bacdive_id: int = 0,
    ):
        self.username = username
        self.password = password
        self.min_bacdive_id = min_bacdive_id
        self.max_bacdive_id = max_bacdive_id
        self.query_list = list(range(self.min_bacdive_id, self.max_bacdive_id, 1))

    def load_credentials(self, filepath: str) -> Tuple[str, str]:
        """Loads secret credentials from file.

        Required file format:
            ```
            username
            password
            ```
        Args:
            filepath: Path to secret credentials file
        Returns:
            username, password: Tuple of username and password
        """
        with open(filepath) as fh:
            username = fh.readline().strip()
            password = fh.readline().strip()
        return username, password

    def scrape_bacdive_api(self) -> dict:
        """Main function for scraping BacDive API

        Returns:
            results: Dictionary keyed by BacDive ID
        """

        logging.info("Logging into BacdiveClient")
        client = bacdive.BacdiveClient(self.username, self.password)

        results = self.paginated_query(
            client=client,
            query_type="id",
        )

        return results

    def paginated_query(self, client, query_type: str) -> Dict[str, dict]:
        """Returns a dictionary keyed by BacDive ID.

        The BacDive API limits to 100 queries per API call. This
        function chunks out a query accordingly.

        Args:
            client: bacdive.BacdiveClient object
            query_type: Query type for API
            query_list: Specific list of queries within a type

        Returns:
            results: Dictionary keyed by BacDive ID
        """
        results = {}
        chunk_size = 100  # BacDive API call limit
        chunks = max([1, round(len(self.query_list) / chunk_size)])
        logging.info("Iniating %s queries in %s chunks", len(self.query_list), chunks)
        for n_split in range(chunks):
            l_idx = chunk_size * n_split
            r_idx = chunk_size * (n_split + 1)
            query = {query_type: self.query_list[l_idx:r_idx]}

            client.result = {}  # Refreshes queue for retrieve
            count = client.search(**query)
            for strain in client.retrieve():
                bacdive_id = strain["General"]["BacDive-ID"]
                results[bacdive_id] = strain
            logging.info("Searching query indices %s-%s returned %s results", l_idx, r_idx, count)

        return results


class ComputeBacDiveTraits:
    """
    Parses BacDive API output to return reported information

    The BacDive API returns, for each BacDive id, a nested dictionary
    with information about the strain. The types of information can be
    found at https://api.bacdive.dsmz.de/strain_fields_information though
    the user should note that the '_' are replaced with ' ' and sections
    are keyed by the section name, not section ID. Multiple sets of values
    may be available for each type of data, in which case values are returned
    as a list of dictionaries instead of a dictionary.

    This class extracts information about strains: either info (taxid and
    genome accession) or conditions (pH, temperature, salinity, oxygen tolerance,
    media). Optimum values are the (average of) the reported optimum(s). Reported
    values refer to conditions in which positive growth was reported, which is for
    many strains a subset of the true range in which they can grow.

    If a version is missing from a GenBank accession, the version '.1' is appended.

    Typical usage:
    ```
    for strain_id, data in bacdive_dict.items():
        strain_traits = ComputeBacDiveTraits(data).compute_trait_data()
    ```
    """

    def __init__(self, entry):
        self.entry = entry

        self.strain_id = self.entry.get("General", {}).get("BacDive-ID", None)
        self.taxid_ncbi = self.get_taxid_ncbi()
        self.species = self.get_species()
        self.genome_accession_ncbi = self.get_genome_accession_ncbi()

        self.reported_media = self.get_reported_media()
        self.reported_temperatures = self.get_reported_temperatures()
        self.reported_phs = self.get_reported_phs()
        self.reported_salinities = self.get_reported_salinities()
        self.reported_oxygen_tolerances = self.get_reported_oxygen_tolerances()

        self.temperature_optimum = self.get_optimum_temperature()
        self.ph_optimum_min, self.ph_optimum_max = self.get_optimum_ph_min_max()
        self.ph_optimum = self.get_optimum_ph(self.ph_optimum_min, self.ph_optimum_max)
        self.salinity_optimum = self.get_optimum_salinity()
        self.salinity_midpoint = self.compute_midpoint_salinity()
        self.oxygen = self.onehot_oxygen_tolerance().get("oxygen", None)

        self.salinity_min = min(self.reported_salinities, default=None)
        self.salinity_max = max(self.reported_salinities, default=None)
        self.ph_min = min(self.reported_phs, default=None)
        self.ph_max = max(self.reported_phs, default=None)
        self.temperature_min = min(self.reported_temperatures, default=None)
        self.temperature_max = max(self.reported_temperatures, default=None)

    def _query_list_of_dicts(self, obj, key, required_key, required_vals: list):
        """Helper to  parsed nested dictionary or list of dictionaries.

        Keys values from 1 dict or a list of dicts with a
        condition that another key-value pair is present
        """
        arr = []
        if isinstance(obj, dict):
            val = self._query_dict_conditionally(obj, key, required_key, required_vals)
            if val:
                arr.append(val)
        elif isinstance(obj, list):  # multiple values
            for subobj in obj:
                if isinstance(subobj, dict):
                    val = self._query_dict_conditionally(subobj, key, required_key, required_vals)
                    if val:
                        arr.append(val)
        return arr

    def _query_dict_conditionally(self, obj: dict, key, required_key, required_vals: list):
        """Looks up value if other value in dictionary is among accepted values"""
        if obj.get(required_key, None) in required_vals:
            return obj.get(key, None)
        else:
            return None

    def _format_values(self, string: str) -> list:
        """
        Uses regex and replace to extract non-float characters from
        strings and correct for typos in data entry. If a range,
        like 3.4-8.4, both values will be returned. Otherwise, one
        value will be returned.
        """
        regex = re.compile(r"[-+]?(?:\d*\.*\d+)")
        if "-" in string:
            return [float(regex.search(val).group(0).replace("..", ".")) for val in string.split("-") if len(val) > 0]
        else:
            return [float(regex.search(string).group(0))]

    def get_reported_media(self) -> set:
        subsection = self.entry.get("Culture and growth conditions", None).get("culture medium", {})
        media_ids = self._query_list_of_dicts(subsection, "@ref", "growth", ["yes", "positive"])
        return set(media_ids)

    def get_genome_accession_ncbi(self) -> str:
        subsection = self.entry.get("Sequence information", {}).get("Genome sequences", {})
        accessions = self._query_list_of_dicts(subsection, "accession", "database", ["ncbi"])
        if len(accessions) > 0:
            accession = accessions[0]
            if len(accession.split(".")) == 1:
                accession += ".1"
            return accession
        else:
            return None

    def get_taxid_ncbi(self) -> str:
        """Returns taxid for lowest taxonomic level"""
        subsection = self.entry.get("General", {}).get("NCBI tax id", {})
        for level in [
            "strain",
            "species",
            "genus",
            "family",
            "order",
            "class",
            "phylum",
            "domain",
        ]:
            taxid = self._query_list_of_dicts(subsection, "NCBI tax id", "Matching level", [level])
            if taxid:
                return taxid[0]
        return None

    def get_species(self) -> str:
        return self.entry.get("Name and taxonomic classification", {}).get("species", None)

    def get_reported_oxygen_tolerances(self) -> set:
        subsection = self.entry.get("Physiology and metabolism", None).get("oxygen tolerance", {})
        tolerances = self._query_list_of_dicts(subsection, "oxygen tolerance", "", [None])
        return set(tolerances)

    def get_reported_temperatures(self) -> set:
        temperatures = []
        subsection = self.entry.get("Culture and growth conditions", {}).get("culture temp", {})
        for val in self._query_list_of_dicts(subsection, "temperature", "growth", ["yes", "positive"]):
            temperatures.extend(self._format_values(val))
        return set(temperatures)

    def get_reported_phs(self) -> set:
        phs = []
        subsection = self.entry.get("Culture and growth conditions", {}).get("culture pH", {})
        for val in self._query_list_of_dicts(subsection, "pH", "ability", ["yes", "positive"]):
            phs.extend(self._format_values(val))
        return set(phs)

    def get_reported_salinities(self) -> set:
        salinities = []
        subsection = self.entry.get("Physiology and metabolism", None).get("halophily", {})
        if isinstance(subsection, dict):
            salinities.extend(self.parse_halophily_dict(subsection))
        elif isinstance(subsection, list):
            for _dict in subsection:
                salinities.extend(self.parse_halophily_dict(_dict))
        return set(salinities)

    def get_optimum_ph_min_max(self) -> tuple:
        phs = []
        subsection = self.entry.get("Culture and growth conditions", {}).get("culture pH", {})
        for val in self._query_list_of_dicts(subsection, "pH", "type", ["optimum"]):
            phs.extend(self._format_values(val))
        if len(phs) > 0:
            return np.min(phs), np.max(phs)
        else:
            return None, None

    def get_optimum_ph(self, min_optimum, max_optimum) -> float:
        if min_optimum and max_optimum:
            return np.mean([min_optimum, max_optimum])
        else:
            return None

    def get_optimum_temperature(self) -> list:
        temperatures = []
        subsection = self.entry.get("Culture and growth conditions", {}).get("culture temp", {})
        for val in self._query_list_of_dicts(subsection, "temperature", "type", ["optimum"]):
            temperatures.extend(self._format_values(val))
        if len(temperatures) > 0:
            return np.mean(temperatures)
        else:
            return None

    def get_optimum_salinity(self) -> list:
        salinities = []
        subsection = self.entry.get("Physiology and metabolism", None).get("halophily", {})
        if isinstance(subsection, dict):
            salinities.extend(self.parse_halophily_dict(subsection, optimum_only=True))
        elif isinstance(subsection, list):
            for _dict in subsection:
                salinities.extend(self.parse_halophily_dict(_dict, optimum_only=True))
        if len(salinities) > 0:
            return np.mean(salinities)
        else:
            return None

    def compute_midpoint_salinity(self) -> float:
        if len(self.reported_salinities) > 0:
            return np.mean([min(self.reported_salinities), max(self.reported_salinities)])

    def parse_halophily_dict(self, halophily: dict, optimum_only=False) -> list:
        """
        Returns growth range of salinity (% NaCl) with the
        following assumptions:

        - No growth > value means value is maximum, minimum is 0
        - Positive growth < value means value is maximum, minimum is 0
        - Other tested relations as point values for positive or inconsistent
            growth but not no growth
        """
        concentrations = []
        concentration = halophily.get("concentration", "")
        growth = halophily.get("growth")
        salt = halophily.get("salt")
        tested_relation = halophily.get("tested relation", None)

        # conversions for NaCl only
        if "g/L" in concentration:
            conversion = 0.1
        elif "M" in concentration:
            conversion = 58.443 / 10.0
        elif "%" in concentration:
            conversion = 1
        else:
            conversion = 1

        if salt == "NaCl":
            if concentration.startswith(">") and growth == "no":
                concentrations = [0.0] + self._format_values(concentration)
            elif concentration.startswith("<") and growth in [
                "positive",
                "inconsistent",
            ]:
                concentrations = [0.0] + self._format_values(concentration)
            elif growth in ["positive", "inconsistent"]:
                concentrations = self._format_values(concentration)

        percent_nacl = [val * conversion for val in concentrations]
        salinities = [val for val in percent_nacl if val < 39]
        if optimum_only:
            if tested_relation == "optimum":
                return salinities
            else:
                return []
        else:
            return salinities

    def _onehot_range(self, arr, min_bin: float, max_bin: float, step: float, prefix: str) -> Dict[str, float]:
        """Return onehot ranges formatted with prefixes"""
        onehot_dict = {}
        for bin_floor in np.arange(min_bin, max_bin, step):
            if min(arr, default=-1000) <= bin_floor <= max(arr, default=-1000):
                onehot_dict[str(bin_floor)] = 1
            else:
                onehot_dict[str(bin_floor)] = 0
        return {f"{prefix}_{k}": v for k, v in onehot_dict.items()}

    def onehot_oxygen_tolerance(self) -> Dict[str, int]:
        """
        Returns a dictionary of oxygen tolerance definitions with 1
        indicating the organism has that tolerance. Oxygen tolerance
        is assigned 1 or 0 based on subtypes and is only set to 0 if
        no aerotolerant subtypes are recorded AND the organism is described
        as an anaerobe or obligate anaerobe.
        """
        onehot_tolerances = {
            "aerobe": None,
            "anaerobe": None,
            "microaerophile": None,
            "facultative anaerobe": None,
            "obligate aerobe": None,
            "obligate anaerobe": None,
            "facultative aerobe": None,
            "aerotolerant": None,
            "microaerotolerant": None,
        }

        aerobes = {
            "obligate aerobe",
            "aerobe",
            "facultative anaerobe",
            "facultative aerobe",
            "microaerophile",
            "aerotolerant",
        }
        obligate_anaerobes = {"obligate anaerobe", "anaerobe"}

        for tolerance in self.reported_oxygen_tolerances:
            onehot_tolerances[tolerance] = 1
        if len(self.reported_oxygen_tolerances.intersection(aerobes)) > 0:
            onehot_tolerances["oxygen"] = 1
        elif len(self.reported_oxygen_tolerances.intersection(obligate_anaerobes)) > 0:
            onehot_tolerances["oxygen"] = 0
        else:
            onehot_tolerances["oxygen"] = None
        return onehot_tolerances

    def feature_quality(self) -> dict:
        """Runs screens to flag data as good (True) or bad (False)"""

        quality_dict = {
            "use_ph": self.qc_condition(
                self.ph_optimum,
                self.ph_min,
                self.ph_max,
                n_values_reported=len(self.reported_phs),
                required_range=1.5,
                keep_below=4,
                keep_above=9,
                optima_to_check=FREQUENT_OPTIMUM_PH,
            ),
            "use_temperature": self.qc_condition(
                self.temperature_optimum,
                self.temperature_min,
                self.temperature_max,
                n_values_reported=len(self.reported_temperatures),
                required_range=10,
                keep_below=19,
                keep_above=45,
                optima_to_check=FREQUENT_OPTIMUM_TEMP,
            ),
            "use_salinity": self.qc_condition(
                self.salinity_optimum,
                self.salinity_min,
                self.salinity_max,
                n_values_reported=len(self.reported_salinities),
                required_range=0.5,
                keep_below=0,  # required to keep 0% salinity values
                keep_above=15,
                optima_to_check=FREQUENT_OPTIMUM_SALINITY,
            ),
            "use_oxygen": self.oxygen is not None,
        }
        return quality_dict

    def qc_condition(
        self,
        optimum: float,
        min_: float,
        max_: float,
        n_values_reported: int,
        required_range: float,
        optima_to_check: list,
        keep_below: float,
        keep_above: float,
    ):
        """Returns False if the optimum is suspect.

        If min and max values are not reported or equal to the optimum,
        it is more likely that the experimenter only tested one condition
        and is therefore not reporting the true optimum.
        """
        use_condition = True
        if optimum is not None and min_ is not None and max_ is not None:
            if optimum in optima_to_check:
                if min_ == optimum or max_ == optimum:
                    use_condition = False
                if n_values_reported < 4:
                    use_condition = False
            # keep extreme optima - more likely to be accurate
            if optimum <= keep_below or optimum >= keep_above:
                use_condition = True
            elif abs(float(max_) - float(min_)) <= required_range:
                use_condition = False
        else:
            use_condition = False

        return use_condition

    def compute_trait_data(self):
        """
        Loads trait data, differently by source, to a set of
        features that will be used for modeling.
        """

        features = {
            "ncbi_accession": self.genome_accession_ncbi,
            "ncbi_taxid": self.taxid_ncbi,
            "strain_id": self.strain_id,
            "species": self.species,
            "ph_optimum": self.ph_optimum,
            "ph_optimum_min": self.ph_optimum_min,
            "ph_optimum_max": self.ph_optimum_max,
            "temperature_optimum": self.temperature_optimum,
            "salinity_optimum": self.salinity_optimum,
            "salinity_midpoint": self.salinity_midpoint,
            "salinity_min": self.salinity_min,
            "salinity_max": self.salinity_max,
            "ph_min": self.ph_min,
            "ph_max": self.ph_max,
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "oxygen": self.oxygen,
        }

        features.update(self.feature_quality())
        features.update(self.onehot_oxygen_tolerance())

        return features


def load_targets_to_dataframe(bacdive_download_file: str) -> pd.DataFrame:
    """Use ComputeBacDiveTraits to create a dataframe
    calculated from the downloaded BacDive data.

    Args:
        bacdive_download_file (str): path to BacDive data downloaded using
            the script download_training_data.py

    Returns:
        df_targets (pd.DataFrame): dataframe with trait data
    """
    bacdive_dict = json.loads(open(bacdive_download_file, "r").read())
    trait_dict = {}
    for data in bacdive_dict.values():
        strain_traits = ComputeBacDiveTraits(data).compute_trait_data()
        genome_accession = strain_traits.get("ncbi_accession", None)
        if genome_accession:
            trait_dict[genome_accession] = strain_traits
    traits_df = pd.DataFrame(trait_dict).T
    return traits_df


def get_bacdive_trait_data(
    bacdive_output: str,
    bacdive_username: str,
    bacdive_password: str,
    max_bacdive_id: int,
    min_bacdive_id: int = 0,
    traits_tsv: str = "trait_data.tsv",
    genome_accessions_txt: str = "genbank_accessions.txt",
    use_existing: bool = False,
):
    """Main function to download and engineer data from BacDive.

    If an existing BacDive download is not provided, this will download
    data from the BacDive API.

    Args:
        output: Filepath to report output trait data
        credentials_file: Personal credentials - see QueryBacDive
        max_bacdive_id: Maximum BacDive id to query - see QueryBacDive
        min_bacdive_id: (Optional) Minimum BacDive id to query - see QueryBacDive
        bacdive_json: (Optional) A preexisting download from BacDive
    """
    # Query BacDive
    if use_existing is not True:
        logging.info("Attempting to download data from BacDive API")
        bacdive_dict = QueryBacDive(
            username=bacdive_username,
            password=bacdive_password,
            max_bacdive_id=int(max_bacdive_id),
            min_bacdive_id=int(min_bacdive_id),
        ).scrape_bacdive_api()
        logging.info("Saving BacDive data to file: %s", bacdive_output)
        json.dump(bacdive_dict, open(bacdive_output, "w"))
    else:
        logging.info("Data from BacDive API supplied by user")
        bacdive_dict = json.loads(open(bacdive_output, "r").read())

    # Compute trait data indexed by genome
    traits_df = load_targets_to_dataframe(bacdive_output)
    traits_df.to_csv(traits_tsv, sep="\t")

    # Save genome list
    with open(genome_accessions_txt, "w") as fh:
        accessions = traits_df.index.tolist()
        fh.write("\n".join(accessions))

    return traits_df


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GetBacDiveTraitData",
        description="Scrapes BacDive API for all strain data and computes traits",
    )
    parser.add_argument(
        "-u",
        "--username",
        help="BacDive username",
    )
    parser.add_argument(
        "-p",
        "--password",
        help="BacDive password",
    )
    parser.add_argument("--min", default=0, help="Lowest BacDive ID to query", required=False)
    parser.add_argument("--max", help="Highest BacDive ID to query")
    parser.add_argument(
        "-b",
        "--bacdive-download",
        default="bacdive_data.json",
        help="Optional: filepath to save raw BacDive data as JSON",
        required=False,
    )
    parser.add_argument("--use-existing", action="store_true", help="Existing  BacDive data download")
    parser.add_argument("-o", "--output", default="trait_data.tsv", help="Output TSV of trait data")
    parser.add_argument(
        "-a", "--accessions", default="genbank_accessions.txt", help="Output TXT file with genome accessions"
    )
    args = parser.parse_args()
    if args.use_existing:
        if not args.bacdive_download:
            raise ValueError("If using existing BacDive data, a filepath must be provided")
    return args


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    args = parse_args()
    df_targets = get_bacdive_trait_data(
        bacdive_output=args.bacdive_download,
        traits_tsv=args.output,
        genome_accessions_txt=args.accessions,
        bacdive_username=args.username,
        bacdive_password=args.password,
        max_bacdive_id=int(args.max),
        min_bacdive_id=int(args.min),
        use_existing=args.use_existing,
    )
