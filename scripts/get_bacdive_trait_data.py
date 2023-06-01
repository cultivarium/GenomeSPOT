#!/usr/bin/env python3

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
from typing import Tuple
import argparse
import json
import logging
import re

import bacdive
import numpy as np

from helpers import onehot_range

class QueryBacDive():
    """Downloads all strain data on the BacDive API.

    Queries all BacDive IDs from 0 to the highest ID available, which 
    as of 2023-03 was about 171000. IDs appear to be sequential.

    Args: 
        credentials_file: Filepath to file with BacDive API credentials. 
            See function`load_credentials`
        min_bacdive_id: Minimum BacDive ID, default of 0 instead of 1 
            for readability
        max_bacdive_id: Highest BacDive ID to query. Set above the highest 
            available ID. See function `find_highest_bacdive_id`
    """
    def __init__(self, 
                 credentials_filepath : str, 
                 max_bacdive_id : int,
                 min_bacdive_id : int = 0,
                ):

        self.username, self.password = self.load_credentials(credentials_filepath)
        self.min_bacdive_id = min_bacdive_id
        self.max_bacdive_id = max_bacdive_id
        self.query_list = list(range(self.min_bacdive_id, self.max_bacdive_id))

    def load_credentials(self, filepath : str) -> Tuple[str, str]:
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
        """ Main function for scraping BacDive API

        Returns:
            results: Dictionary keyed by BacDive ID
        """

        logging.info('Logging into BacdiveClient')
        client = bacdive.BacdiveClient(self.username, self.password)

        results = self.paginated_query(
                    client=client,
                    query_type ='id',
                    )
        
        del username
        del password

        return results

    def paginated_query(self,
            client,
            query_type : str) -> dict:

        """ Returns a dictionary keyed by BacDive ID.

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
        chunk_size = 100 # BacDive API call limit
        chunks = round(len(self.query_list)/chunk_size)
        logging.info('Iniating {} queries in {} chunks'.format(len(self.query_list), chunks))
        for n_split in range(chunks):
            
            l_idx = chunk_size * n_split
            r_idx = chunk_size * (n_split + 1)
            query = {query_type: self.query_list[l_idx:r_idx]}

            client.result = {} # Refreshes queue for retrieve
            count = client.search(**query)
            for strain in client.retrieve():
                bacdive_id = strain['General']['BacDive-ID']
                results[bacdive_id] = strain
            logging.info( "Searching query indices {}-{} returned {} results".format(l_idx, r_idx, count))

        return results


class BacDiveData():
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
    """
    
    def __init__(self, entry):
        
        self.entry = entry

        self.strain_id = self.entry.get('General', {}).get( 'BacDive-ID', None)
        self.taxid_ncbi = self.get_taxid_ncbi()
        self.genome_accession_ncbi = self.get_genome_accession_ncbi()

        self.reported_media = self.get_reported_media()
        self.reported_temperatures = self.get_reported_temperatures()
        self.reported_phs = self.get_reported_phs()
        self.reported_salinities = self.get_reported_salinities()
        self.reported_oxygen_tolerances = self.get_reported_oxygen_tolerances()
        
        self.optimum_ph = self.get_optimum_ph()
        self.optimum_temperature = self.get_optimum_temperature()
        self.midpoint_salinity = self.compute_midpoint_salinity()
    
    def get_reported_media(self) -> list:
        subsection = self.entry.get('Culture and growth conditions', None).get('culture medium', {})
        media_ids = self._query_list_of_dicts(subsection, '@ref', 'growth', ['yes', 'positive'])
        return set(media_ids)
    
    def get_reported_temperatures(self) -> list:
        temperatures = []
        subsection = self.entry.get('Culture and growth conditions', {}).get('culture temp', {})
        for val in self._query_list_of_dicts(subsection, 'temperature', 'growth', ['yes', 'positive']):
            temperatures.extend(self._format_values(val))
        return set(temperatures)

    def get_reported_phs(self) -> list:
        phs = []
        subsection = self.entry.get('Culture and growth conditions', {}).get('culture pH', {})
        for val in self._query_list_of_dicts(subsection, 'pH', 'ability', ['yes', 'positive']):
            phs.extend(self._format_values(val))
        return set(phs)

    def get_reported_salinities(self) -> list:
        salinities = []
        subsection = self.entry.get('Physiology and metabolism', None).get('halophily', {})
        if isinstance(subsection, dict):
            salinities.extend(self.parse_halophily_dict(subsection))
        elif isinstance(subsection, list):
            for _dict in subsection:
                salinities.extend(self.parse_halophily_dict(_dict))
        return set(salinities)
    
    def get_optimum_ph(self) -> list:
        phs = []
        subsection = self.entry.get('Culture and growth conditions', {}).get('culture pH', {})
        for val in self._query_list_of_dicts(subsection, 'pH', 'type', ['optimum']):
            phs.extend(self._format_values(val))
        if len(phs) > 0:
            return np.mean(phs)
        else:
            return None
    
    def get_optimum_temperature(self) -> list:
        temperatures = []
        subsection = self.entry.get('Culture and growth conditions', {}).get('culture temp', {})
        for val in self._query_list_of_dicts(subsection, 'temperature',  'type', ['optimum']):
            temperatures.extend(self._format_values(val))
        if len(temperatures) > 0:
            return np.mean(temperatures)
        else:
            return None
        
    def get_reported_oxygen_tolerances(self) -> list:
        subsection = self.entry.get('Physiology and metabolism', None).get('oxygen tolerance', {})
        tolerances = self._query_list_of_dicts(subsection, 'oxygen tolerance', '', [None])
        return set(tolerances)
    
    def get_genome_accession_ncbi(self) -> str:
        subsection = self.entry.get('Sequence information', {}).get('Genome sequences', {})
        accessions = self._query_list_of_dicts(subsection, 'accession',  'database', ['ncbi'])
        if len(accessions) > 0:
            return accessions[0]
        else:
            return None
        
    def get_taxid_ncbi(self) -> str:
        """Returns taxid for lowest taxonomic level"""
        subsection = self.entry.get('General', {}).get('NCBI tax id', {})
        for level in ['strain', 'species', 'genus', 'family', 'order', 'class', 'phylum', 'domain']:
            taxid = self._query_list_of_dicts(subsection, 'NCBI tax id',  'Matching level', [level])
            if taxid:
                return taxid[0]
        return None
    
    def compute_midpoint_salinity(self):
        if len(self.reported_salinities) > 0:
            return np.mean([min(self.reported_salinities), max(self.reported_salinities)])

    def parse_halophily_dict(self, halophily : dict):
        """
        Returns growth range of salinity (% NaCl) with the 
        following assumptions:
        
        - No growth > value means value is maximum, minimum is 0
        - Positive growth < value means value is maximum, minimum is 0
        - Other tested relations as point values for positive or inconsistent
            growth but not no growth
        """
        concentrations = []
        concentration = halophily.get('concentration', '')
        growth = halophily.get('growth')
        salt = halophily.get('salt')
        
        # conversions for NaCl only
        if 'g/L' in concentration:
            conversion = 0.1
        elif 'M' in concentration:
            conversion = 58.443 / 10.
        elif '%' in concentration:
            conversion = 1
        else:
            conversion = 1
            
        if salt == 'NaCl':
            if concentration.startswith('>') and growth == 'no':
                concentrations = [0.] + self._format_values(concentration)
            elif concentration.startswith('<') and growth in ['positive', 'inconsistent']:
                concentrations = [0.] + self._format_values(concentration)
            elif growth in ['positive', 'inconsistent']:
                concentrations = self._format_values(concentration)

        percent_nacl = [val * conversion for val in concentrations] 
        return [val for val in percent_nacl if val < 39]
    
    def _query_list_of_dicts(self, obj, key, required_key, required_vals : list):
        """
        Keys values from 1 dict or a list of dicts with a 
        condition that another key-value pair is present
        """
        arr = []
        if isinstance(obj, dict):
            val = self._query_dict_conditionally(obj, key, required_key, required_vals)
            if val:
                arr.append(val)
        elif isinstance(obj, list): # multiple values
            for subobj in obj:
                if isinstance(subobj, dict):
                    val = self._query_dict_conditionally(subobj, key, required_key, required_vals)
                    if val:
                        arr.append(val)
        return arr
    
    def _query_dict_conditionally(self, obj : dict, key, required_key, required_vals : list):
        """Lookup value if other value in dictionary among accepted values"""
        if obj.get(required_key, None) in required_vals:
            return obj.get(key, None)
        else:
            return None

    def _format_values(self, string : str) -> list:
        """
        Uses regex and replace to extract non-float characters from
        strings and correct for typos in data entry. If a range,
        like 3.4-8.4, both values will be returned. Otherwise, one
        value will be returned.
        """
        regex = re.compile(r"[-+]?(?:\d*\.*\d+)")
        if '-' in string:
            return [float(regex.search(val).group(0).replace('..', '.')) for val in string.split('-') if len(val) > 0]
        else:
            return [float(regex.search(string).group(0))]
        
    def _onehot_range(arr, min_bin : float, max_bin : float, step : float, prefix : str) -> dict:
        """Return onehot ranges formatted with prefixes"""
        onehot_dict = onehot_range(arr=arr, min_bin=min_bin, max_bin=max_bin, step=step)
        return {f"{prefix}_{k}" : v for k, v in onehot_dict.items()}
    
    def onehot_oxygen_tolerance(self, tolerances : set) -> dict:
        """
        Returns a dictionary of oxygen tolerance definitions with 1
        indicating the organism has that tolerance. Subtypes of aerobe 
        and anaerobe lead to assignment of 1 to aerobe or anaerobe, 
        respectivey, with facultative anaerobes only assigned to aerobe. 
        """
        onehot_tolerances = {'aerobe' : None, 
                       'anaerobe' : None, 
                       'microaerophile' : None , 
                       'facultative anaerobe' : None, 
                       'obligate aerobe' : None, 
                       'obligate anaerobe' : None, 
                       'facultative aerobe' : None, 
                       'aerotolerant' : None, 
                       'microaerotolerant' : None}
        
        aerobe_subtypes = {'facultative anaerobe', 'obligate aerobe', 'facultative aerobe', 'microaerophile'}
        anaerobe_subtypes = {'obligate anaerobe', 'facultative aerobe'}
        
        for tolerance in tolerances:
            onehot_tolerances[tolerance] = 1
        if len(tolerances.intersection(aerobe_subtypes)) > 0:
            onehot_tolerances['aerobe'] = 1
        if len(tolerances.intersection(anaerobe_subtypes)) > 0:
            onehot_tolerances['anaerobe'] = 1
        
        return onehot_tolerances
    
    def compute_trait_data(self,):
        """
        Loads trait data, differently by source, to a set of
        features that will be used for modeling.
        """

        features = {
            'ncbi_accession' : self.genome_accession_ncbi,
            'ncbi_taxid' : self.taxid_ncbi,
            'strain_id' : self.strain_id,
            'ph_optimum' : self.optimum_ph,
            'temperature_optimum' : self.optimum_temperature,
            'salinity_midpoint' : self.midpoint_salinity,
            'salinity_min' : min(self.reported_salinities, default=None),
            'salinity_max' : max(self.reported_salinities, default=None),
            'ph_min': min(self.reported_phs, default=None),
            'ph_max': max(self.reported_phs, default=None),
            'temperature_min' : min(self.reported_temperatures, default=None),
            'temperature_max' : max(self.reported_temperatures, default=None),
                   }

        features.update(self._onehot_range(self.reported_salinities, 0, 38.4, 0.5, prefix='nacl')) # salinity range
        features.update(self._onehot_range(self.reported_phs, 0, 14, 0.25, prefix='ph')) # pH range 
        features.update(self._onehot_range(self.reported_temperatures, 0, 100, 1, prefix='temp')) # temperature range
        features.update(self.onehot_oxygen_tolerance(self.reported_oxygen_tolerances)) # o2 tolerance

        return features
    

def get_bacdive_trait_data(output : str,
                                      credentials_file : str ,
                                      max_bacdive_id : int,
                                      min_bacdive_id : int = 0,
                                      bacdive_json : dict = None,
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
    if bacdive_json is None:
        logging.info('Attempting to download data from BacDive API')
        bacdive_query = QueryBacDive(credentials_file = credentials_file, 
                                        max_bacdive_id=int(max_bacdive_id),
                                        min_bacdive_id=int(min_bacdive_id),
                                        )

        bacdive_dict = bacdive_query.scrape_bacdive_api()        
        logging.info('Saving BacDive data to file: {}'.format(bacdive_json)) 
        json.dump(bacdive_dict, open(bacdive_json, 'w'))
    else:
        logging.info('Data from BacDive API supplied by user') 
        bacdive_dict = json.loads(open(bacdive_json).read())

    # Compute trait data indexed by genome
    trait_dict = {}
    for n, (strain_id, data) in enumerate(bacdive_dict.items()):
        strain_traits = BacDiveData(data).compute_trait_data()
        genome_accession = strain_traits.get('ncbi_accession', None)
        if genome_accession:
            trait_dict[genome_accession] = strain_traits

    json.dump(trait_dict, open(output, 'w'))
    return trait_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='QueryBacdive',
                    description='Scrapes BacDive API for all strain data'
                    )
    
    parser.add_argument('-c', '--credentials', help='File with BacDive credentials as: 1st line username, 2nd line password')
    parser.add_argument('-min', default=0, help='Lowest BacDive ID to query', required=False)
    parser.add_argument('-max', help='Highest BacDive ID to query')
    parser.add_argument('-o', '--output', default='bacdive_data.json', help='Output JSON name')

    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", encoding='utf-8', level=logging.INFO)
    
    get_bacdive_trait_data(output=args.output, 
                            credentials_file = args.credentials,
                            max_bacdive_id=int(args.max),
                            min_bacdive_id=int(args.min),
                            output = args.output,
                            bacdive_json = args.bacdive_json,
                            )