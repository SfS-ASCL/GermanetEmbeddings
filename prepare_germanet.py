import argparse
from germanetpy.germanet import Germanet
from germanetpy.lexunit import LexRel
from tqdm import trange


def load_germanet(data_path):
    """
    Loads GermaNet
    :param data_path: the path to the XML data that contain GermaNet
    :return: the GermaNet object
    """
    germanet = Germanet(data_path)
    return germanet


def extract_related(germanet, add_hyponym, add_hypernym):
    """
    for each lexical unit in GermaNet extract and store all related words. which related words will be used can be
    specified by the flags. as a default, only synonyms will be added as related words
    :param germanet: the GermaNet object
    :param add_hyponym: a flag, if True adds all hyponyms of a lexical unit as related lexunits
    :param add_hypernym: a flag, if True adds all hypernyms of a lexical unit as related lexunits
    :return: a dictionary, containing all lexical units and a list of corresponding related lexical units
    """
    lexunits = germanet.lexunits
    lexunit2related = {}
    pbar = trange(len(lexunits), desc='extract related words...', leave=True)
    for id, unit in lexunits.items():
        pbar.update(1)
        word_form = unit.orthform + "_" + id
        related_lexunits = unit.relations
        synonyms = related_lexunits[LexRel.has_synonym]
        synonyms = [syn.orthform + "_" + syn.id for syn in synonyms]
        related_words = set(synonyms)
        synset = unit._synset
        if add_hypernym:
            hyper_synsets = synset.direct_hypernyms
            for hyper_synset in hyper_synsets:
                if hyper_synset.id == "s51001":
                    continue
                hyper_lexunits = hyper_synset.lexunits
                for hyper_lexunit in hyper_lexunits:
                    related_words.add(hyper_lexunit.orthform + "_" + hyper_lexunit.id)
        if add_hyponym:
            hypo_synsets = synset.direct_hyponyms
            for hypo_synset in hypo_synsets:
                hypo_lexunits = hypo_synset.lexunits
                for hypo_lexunit in hypo_lexunits:
                    related_words.add(hypo_lexunit.orthform + "_" + hypo_lexunit.id)

        lexunit2related[word_form] = related_words
    return lexunit2related


def write_to_file(lexunit2related, outfile, sep):
    """
    This method writes the dictionary to an output file
    :param lexunit2related: the dictionary containing all lexical units and related lexical units
    :param outfile: a path that specifies where the dictionary will be stored to
    :param sep: preferred separator for the output file (this will contain two columns: lexunit[sep]relatedLexUnits
    """
    f = open(outfile, "w")
    for unit, related in lexunit2related.items():
        f.write(unit + sep)
        for rel in related:
            f.write(rel + ";")
        f.write("\n")
    f.close()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("germanet_data", type=str, help="the directory that points to the GermaNet data")
    argp.add_argument("outpath", type=str, help="the filename of the LexUnit2RelatedLexUnit dictionary")
    argp.add_argument("sep", help="the separator that separates the two columns of the new file")
    argp.add_argument("--hypernyms", default=False, action='store_true',
                      help="if hypernyms should be added as related lexunits")
    argp.add_argument("--hyponyms", default=False, action='store_true',
                      help="if hyponyms should be added as related lexunits")
    argp = argp.parse_args()

    g = load_germanet(argp.germanet_data)
    lex2related = extract_related(g, add_hypernym=argp.hypernyms, add_hyponym=argp.hyponyms)
    write_to_file(lex2related, argp.outpath, argp.sep)
