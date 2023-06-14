import argparse
import json
from pathlib import Path
import multiprocessing
import logging

from measure_genome import Genome

ORGANISM_TYPE_DICT = './data/references/organism_types.json'
DEFAULT_TYPE = 'gramp'

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

def mapping_wrapper(inputs):
    n, faa_path, fna_path, organism_type, output_path = inputs
    try:
        genome_features = Genome(contig_filepath=fna_path, protein_filepath=faa_path, organism_type=organism_type).genome_metrics()
        json.dump(genome_features, open(output_path, 'w'))
        logger.info("{}: {}".format(n, output_path))
        return output_path
    except:
        return None

def create_mapping_inputs(directory, suffix_fna, suffix_faa, output_dir) -> list:
    pathlist = []
    missing_faas = []
    type_dict = json.loads(open(ORGANISM_TYPE_DICT).read())

    logger.info('Looking for {} and {} files in {}'.format(suffix_fna, suffix_faa, directory))
    fna_pathlists = {genome_accession_from_fasta_path(str(path)) : path for path in Path(directory).rglob('*{}'.format(suffix_fna))}
    faa_pathlists = {genome_accession_from_fasta_path(str(path)) : path for path in Path(directory).rglob('*{}'.format(suffix_faa))}
    for genome, fna_path in fna_pathlists.items():
        faa_path = faa_pathlists.get(genome, None)
        if faa_path:
            genome_accession = genome_accession_from_fasta_path(str(fna_path))
            output_path = "{}/{}.features.json".format(output_dir, genome_accession)
            organism_type = type_dict.get(genome_accession, DEFAULT_TYPE)
            #if Path(output_path).exists():
            #    pass
            #else:
            pathlist.append((faa_path, fna_path, organism_type, output_path))
        else:
            missing_faas.append(fna_path)

    logger.info('Found files for {} genomes'.format(len(pathlist)))
    logger.info('MISSING protein FASTA files for {} genomes'.format(len(missing_faas)))
    return pathlist

def genome_accession_from_fasta_path(path : str):
    """Assumes genome accession is format like:
    `path/to/ALPHA_NUMERIC.#_OTHER_INFORMATION.blah`
    """
    filename = path.split('/')[-1]
    accession = filename.split('.')[0] + filename.split('.')[1].split('_')[0]
    return accession

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='RunGenomeMeasurements',
                    description='Run measure genomes on many genomes in parallel'
                    )
    
    parser.add_argument('-d', '--directory', help='Path to directory containing subdirectories each with one genome FASTA and one protein FASTA')
    parser.add_argument('-sfaa', default='.faa.gz', help='Suffix of protein FASTA files, default .faa.gz')
    parser.add_argument('-sfna', default='.fna.gz', help='Suffix of genome FASTA files, default .fna.gz')
    parser.add_argument('-o', '--output-dir',  help='Output directory to populate with files <genome_accession>.features.json')
    parser.add_argument('-p', default=4, help='Number of parallel processes', required=False)

    args = parser.parse_args()

    input_list = create_mapping_inputs(args.directory, suffix_fna=args.sfna, suffix_faa=args.sfaa, output_dir=args.output_dir)
    
    workers = int(args.p)
    if workers is None:
        workers = multiprocessing.cpu_count() - 1

    logging.info("Measuring {} genomes with {} CPUs".format(len(input_list), workers))
    filepath_gen = (
        (n, Path(faa_path), Path(fna_path), organism_type, output_path) 
        for n, (faa_path, fna_path, organism_type, output_path) in enumerate(input_list)
        )
    with multiprocessing.Pool(workers) as p:
        pipeline_gen = p.map(mapping_wrapper, filepath_gen)
        genomes = list(pipeline_gen)
    logger.info("Measured {} genomes".format(len(genomes)))
    