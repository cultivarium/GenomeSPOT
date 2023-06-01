#!/usr/bin/env python3

import argparse
import json
import logging
from typing import Tuple

import bacdive

def scrape_bacdive_api(
        credentials_file : str,
        max_bacdive_id : int,
        min_bacdive_id : int = 0,
        ):
    """
    **BEFORE USING**: For the BacDive team's sanity,
    please check to see if they have since made a simple way to
    download all data e.g. by FTP.
    
    Downloads all strain data on the BacDive API by querying
    all BacDive IDs from 0 to the highest ID available, which 
    as of 2023-03 was about 171000. IDs appear to be sequential.

    :param credentials_file: str
        Filepath to file with BacDive API credentials. See function
        `load_credentials`
    :param min_bacdive_id: int
        Minimum BacDive ID, default of 0 instead of 1 for readability
    :param max_bacdive_id: int
        Highest BacDive ID to query. Set above the highest 
        available ID. See function `find_highest_bacdive_id`
    """

    username, password = load_credentials(credentials_file)

    logging.info('Logging into BacdiveClient')
    client = bacdive.BacdiveClient(username, password)

    results = paginated_query(
                client=client,
                query_type ='id',
                query_list=list(range(min_bacdive_id, max_bacdive_id)),
                )
    
    del username
    del password

    return results

def paginated_query(
        client,
        query_type : str,
        query_list : list) -> dict:

    """
    Returns a dictionary keyed by BacDive ID.

    The BacDive API limits to 100 queries per API call. This
    function chunks out a query accordingly.



    """
    results = {}
    chunk_size = 100 # BacDive API call limit
    chunks = round(len(query_list)/chunk_size)
    logging.info('Iniating {} queries in {} chunks'.format(len(query_list), chunks))
    for n_split in range(chunks):
        
        l_idx = chunk_size * n_split
        r_idx = chunk_size * (n_split + 1)
        query = {query_type: query_list[l_idx:r_idx]}

        client.result = {} # Refreshes queue for retrieve
        count = client.search(**query)
        for strain in client.retrieve():
            bacdive_id = strain['General']['BacDive-ID']
            results[bacdive_id] = strain
        logging.info( "Searching query indices {}-{} returned {} results".format(l_idx, r_idx, count))

    return results

def load_credentials(filepath : str) -> Tuple[str, str]:
    """
    Loads secret credentials from file in format:
    ```
    username
    password
    ```
    """
    with open(filepath) as fh:
        username = fh.readline().strip()
        password = fh.readline().strip()
    return username, password


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
    
    results = scrape_bacdive_api(credentials_file = args.credentials, 
                                 max_bacdive_id=int(args.max),
                                 min_bacdive_id=int(args.min),
                                )
    json.dump(results, open(args.output, 'w'))