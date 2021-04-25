from nltk.corpus import wordnet as wn
import seaborn as sn
import matplotlib.pyplot as plt
import os, json
import pandas as pd
import json
import traceback
from tqdm import tqdm
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
# lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
pjoin = os.path.join


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def get_pos_preds(preds, pos, vad_df, ap_df=None, lemmatizer=None):
    pos_preds = {}

    for k, v in preds.items():
        word = v[0]
        # skip words with digits and one letter words
        if len(word) < 2 and any(i.isdigit() for i in word):
            continue

        if wn.morphy(word, pos):
            vad_score = get_vad_scores(vad_df, word, lemmatizer)
            new_v = make_vad_pred_dict(v, vad_score)  # add vad to dict
            if ap_df is not None:
                ap_score = get_agency_power_scores(ap_df, word, lemmatizer)  # get ap values
                new_v = make_agency_power_pred_dict(new_v, ap_score)  # add ap to dict
            pos_preds[int(k)] = new_v  # convert string key to int key for sorting

    pos_preds = sorted(pos_preds.items(), key=lambda kv: kv[0])

    return pos_preds


def make_agency_power_pred_dict(d, ap):
    d["agency"] = ap[0]
    d["power"] = ap[1]

    return d


def get_agency_power_scores(ap_df, word, lemmatizer):
    lemma = lemmatizer.verb(u'{}'.format(word))[0]

    if lemma in ap_df.index:

        return list(ap_df.loc[lemma])[1:]
    else:
        return ["", ""]


def make_vad_pred_dict(v, vad):
    d = {}

    d["word"] = v[0]
    d["percent"] = v[1]
    d["weight"] = v[2]
    d["valence"] = vad[0]
    d["arousal"] = vad[1]
    d["dominance"] = vad[2]

    return d


def get_vad_scores(vad_df, word, lemmatizer=None):
    if word in vad_df.index:
        return [round(v, 4) for v in vad_df.loc[word]]
    else:
        if lemmatizer:
            lemma = lemmatizer.verb(u'{}'.format(word))[0]
            if lemma in vad_df.index:
                return [round(v, 4) for v in vad_df.loc[lemma]]
            else:
                return [0.0, 0.0, 0.0]
        else:
            return [0.0, 0.0, 0.0]


def get_pos(typ):
    if typ == "participant":
        pos = wn.NOUN
    elif typ == "adjectives":
        pos = wn.ADJ
    elif typ == "verb":
        pos = wn.VERB
    elif typ == "pronouns":
        pos = "pronoun"
    else:
        pos = ""

    return pos


def add_top_verbs(pos_preds, verb_dict):
    for i, d in pos_preds:
        word = d["word"]
        count = d.get("count", 0) + 1
        valence = d.get("valence", 0.0)
        arousal = d.get("arousal", 0.0)
        dominance = d.get("dominance", 0.0)
        agency = d.get("agency", "NaN")
        power = d.get("power", "NaN")
        percent = d.get("percent", 0.00)

        verb_dict[word] = verb_dict.get(word, {})
        verb_dict[word]["count"] = verb_dict[word].get("count", 0) + count
        verb_dict[word]["percent"] = verb_dict[word].get("percent", []) + [percent]
        verb_dict[word]["valence"] = verb_dict[word].get("valence", valence)
        verb_dict[word]["arousal"] = verb_dict[word].get("arousal", arousal)
        verb_dict[word]["dominance"] = verb_dict[word].get("dominance", dominance)
        verb_dict[word]["agency"] = verb_dict[word].get("agency", agency)
        verb_dict[word]["power"] = verb_dict[word].get("power", power)

    return verb_dict


def convert_preds(preds, typ=None):
    pos_preds = {}

    for k, v in preds.items():
        word = v[0]
        if typ != "pronouns":
            # skip words with digits and one letter words
            if len(word) > 1 and not any(i.isdigit() for i in word):
                continue

        attributes = {}
        attributes["word"] = word
        attributes["percent"] = v[1]
        attributes["weight"] = v[2]

        pos_preds[int(k)] = attributes

    pos_preds = sorted(pos_preds.items(), key=lambda kv: kv[0])

    return pos_preds


def get_template_stats(stats, pos_preds, pred_type, adjective, verb, occupation, participant, answer, pronoun, someone,
                       opp):
    if someone:
        someone_key = "someone"
    else:
        someone_key = "not_someone"

    stats[someone_key] = stats.get(someone_key, {})

    for i, d in pos_preds:
        _word = d["word"]
        _percent = d["weight"] if pred_type == "pronouns" else d["percent"]

        if pred_type == "verb":
            ap = {"p": d["power"], "a": d["agency"]}

        if pred_type in ["adjectives", "verb"]:
            vad = {"valence": d["valence"], "arousal": d["arousal"], "dominance": d["dominance"]}

        if pred_type == "adjectives":
            #             continue
            if not someone and pronoun not in ["they", "their", "them"] or someone:
                stats[someone_key]["adjectives"] = stats[someone_key].get("adjectives", {})
                stats[someone_key]["adjectives"][_word] = stats[someone_key]["adjectives"].get(_word, {})
                stats[someone_key]["adjectives"][_word]["count"] = stats[someone_key]["adjectives"][_word].get("count",
                                                                                                               0) + 1
                stats[someone_key]["adjectives"][_word]["percent"] = stats[someone_key]["adjectives"][_word].get(
                    "percent", []) + [_percent]
                stats[someone_key]["adjectives"][_word]["vad"] = stats[someone_key]["adjectives"][_word].get("vad", vad)

                stats[someone_key]["occupations"] = stats[someone_key].get("occupations", {})
                stats[someone_key]["occupations"][occupation] = stats[someone_key]["occupations"].get(occupation, {})
                stats[someone_key]["occupations"][occupation][_word] = stats[someone_key]["occupations"][
                    occupation].get(_word, {})
                stats[someone_key]["occupations"][occupation][_word]["count"] = \
                stats[someone_key]["occupations"][occupation][_word].get("count", 0) + 1
                stats[someone_key]["occupations"][occupation][_word]["percent"] = \
                stats[someone_key]["occupations"][occupation][_word].get("percent", []) + [_percent]
                stats[someone_key]["occupations"][occupation][_word]["vad"] = \
                stats[someone_key]["occupations"][occupation][_word].get("vad", vad)
                stats[someone_key]["occupations"][occupation][_word]["ap"] = \
                stats[someone_key]["occupations"][occupation][_word].get("ap", ap)



        elif pred_type == "verb":
            # continue
            # skip sentences without adjective if someone (these are still included with adj="nan" in the directory for sentences with adjectives)
            if someone and opp and answer == 1 or someone and not opp and answer == 0 or not someone:
                stats[someone_key]["verbs"] = stats[someone_key].get("verbs", {})
                stats[someone_key]["verbs"][_word] = stats[someone_key]["verbs"].get(_word, {})
                stats[someone_key]["verbs"][_word]["count"] = stats[someone_key]["verbs"][_word].get("count", 0) + 1
                stats[someone_key]["verbs"][_word]["percent"] = stats[someone_key]["verbs"][_word].get("percent",
                                                                                                       []) + [_percent]
                stats[someone_key]["verbs"][_word]["vad"] = stats[someone_key]["verbs"][_word].get("vad", vad)
                stats[someone_key]["verbs"][_word]["ap"] = stats[someone_key]["verbs"][_word].get("ap", ap)

                stats[someone_key]["adjectives"] = stats[someone_key].get("adjectives", {})
                stats[someone_key]["adjectives"][adjective] = stats[someone_key]["adjectives"].get(adjective, {})
                stats[someone_key]["adjectives"][adjective][_word] = stats[someone_key]["adjectives"][adjective].get(
                    _word, {})
                stats[someone_key]["adjectives"][adjective][_word]["count"] = \
                stats[someone_key]["adjectives"][adjective][_word].get("count", 0) + 1
                stats[someone_key]["adjectives"][adjective][_word]["percent"] = \
                stats[someone_key]["adjectives"][adjective][_word].get("percent", []) + [_percent]
                stats[someone_key]["adjectives"][adjective][_word]["vad"] = stats[someone_key]["adjectives"][adjective][
                    _word].get("vad", vad)
                stats[someone_key]["adjectives"][adjective][_word]["ap"] = stats[someone_key]["adjectives"][adjective][
                    _word].get("ap", ap)

                stats[someone_key]["occupations"] = stats[someone_key].get("occupations", {})
                stats[someone_key]["occupations"][occupation] = stats[someone_key]["occupations"].get(occupation, {})
                stats[someone_key]["occupations"][occupation][_word] = stats[someone_key]["occupations"][
                    occupation].get(_word, {})
                stats[someone_key]["occupations"][occupation][_word]["count"] = \
                stats[someone_key]["occupations"][occupation][_word].get("count", 0) + 1
                stats[someone_key]["occupations"][occupation][_word]["percent"] = \
                stats[someone_key]["occupations"][occupation][_word].get("percent", []) + [_percent]
                stats[someone_key]["occupations"][occupation][_word]["vad"] = \
                stats[someone_key]["occupations"][occupation][_word].get("vad", vad)
                stats[someone_key]["occupations"][occupation][_word]["ap"] = \
                stats[someone_key]["occupations"][occupation][_word].get("ap", ap)

                if not someone:
                    stats[someone_key]["participants"] = stats[someone_key].get("participants", {})
                    stats[someone_key]["participants"][participant] = stats[someone_key]["participants"].get(
                        participant, {})
                    stats[someone_key]["participants"][participant][_word] = stats[someone_key]["participants"][
                        participant].get(_word, {})
                    stats[someone_key]["participants"][participant][_word]["count"] = \
                    stats[someone_key]["participants"][participant][_word].get("count", 0) + 1
                    stats[someone_key]["participants"][participant][_word]["percent"] = \
                    stats[someone_key]["participants"][participant][_word].get("percent", []) + [_percent]
                    stats[someone_key]["participants"][participant][_word]["vad"] = \
                    stats[someone_key]["participants"][participant][_word].get("vad", vad)
                    stats[someone_key]["participants"][participant][_word]["ap"] = \
                    stats[someone_key]["participants"][participant][_word].get("ap", ap)

        elif pred_type == "pronouns":
            # continue
            # skip sentences without adjective if someone (these are still included with adj="nan" in the directory for sentences with adjectives)
            if someone and opp and answer == 1 or someone and not opp and answer == 0 or not someone:
                #                 print(adjective)
                stats[someone_key]["pronouns"] = stats[someone_key].get("pronouns", {})
                stats[someone_key]["pronouns"][_word] = stats[someone_key]["pronouns"].get(_word, {})
                stats[someone_key]["pronouns"][_word]["count"] = stats[someone_key]["pronouns"][_word].get("count",
                                                                                                           0) + 1
                stats[someone_key]["pronouns"][_word]["percent"] = stats[someone_key]["pronouns"][_word].get("percent",
                                                                                                             []) + [
                                                                       _percent]

                stats[someone_key]["adjectives"] = stats[someone_key].get("adjectives", {})
                stats[someone_key]["adjectives"][adjective] = stats[someone_key]["adjectives"].get(adjective, {})
                stats[someone_key]["adjectives"][adjective][_word] = stats[someone_key]["adjectives"][adjective].get(
                    _word, {})
                stats[someone_key]["adjectives"][adjective][_word]["count"] = \
                stats[someone_key]["adjectives"][adjective][_word].get("count", 0) + 1
                stats[someone_key]["adjectives"][adjective][_word]["percent"] = \
                stats[someone_key]["adjectives"][adjective][_word].get("percent", []) + [_percent]

                # when occuaption noun is coreferent of pronoun
                if answer == 0 and not opp:
                    stats[someone_key]["occupations"] = stats[someone_key].get("occupations", {})
                    stats[someone_key]["occupations"][occupation] = stats[someone_key]["occupations"].get(occupation,
                                                                                                          {})
                    stats[someone_key]["occupations"][occupation][adjective] = stats[someone_key]["occupations"][
                        occupation].get(adjective, {})
                    stats[someone_key]["occupations"][occupation][adjective][_word] = \
                    stats[someone_key]["occupations"][occupation][adjective].get(_word, {})
                    stats[someone_key]["occupations"][occupation][adjective][_word]["count"] = \
                    stats[someone_key]["occupations"][occupation][adjective][_word].get("count", 0) + 1
                    stats[someone_key]["occupations"][occupation][adjective][_word]["percent"] = \
                    stats[someone_key]["occupations"][occupation][adjective][_word].get("percent", []) + [_percent]

                # when participant noun is coreferent of pronoun
                if answer == 1 and opp:
                    stats[someone_key]["participants"] = stats[someone_key].get("participants", {})
                    stats[someone_key]["participants"][participant] = stats[someone_key]["participants"].get(
                        participant, {})
                    stats[someone_key]["participants"][participant][adjective] = stats[someone_key]["participants"][
                        participant].get(adjective, {})
                    stats[someone_key]["participants"][participant][adjective][_word] = \
                    stats[someone_key]["participants"][participant][adjective].get(_word, {})
                    stats[someone_key]["participants"][participant][adjective][_word]["count"] = \
                    stats[someone_key]["participants"][participant][adjective][_word].get("count", 0) + 1
                    stats[someone_key]["participants"][participant][adjective][_word]["percent"] = \
                    stats[someone_key]["participants"][participant][adjective][_word].get("percent", []) + [_percent]

                stats[someone_key]["verbs"] = stats[someone_key].get("verbs", {})
                stats[someone_key]["verbs"][verb] = stats[someone_key]["verbs"].get(verb, {})
                stats[someone_key]["verbs"][verb][_word] = stats[someone_key]["verbs"][verb].get(_word, {})
                stats[someone_key]["verbs"][verb][_word]["count"] = stats[someone_key]["verbs"][verb][_word].get(
                    "count", 0) + 1
                stats[someone_key]["verbs"][verb][_word]["percent"] = stats[someone_key]["verbs"][verb][_word].get(
                    "percent", []) + [_percent]

    return stats


def generate_vad_ap_preds_and_stats(unique_dirs, save_path, top_k=None):
    # preds statistics
    stats = {}

    data_dir = pjoin(os.getcwd(), os.pardir, "data")
    vad_file = pjoin(data_dir, "NRC-VAD-Lexicon", "NRC-VAD-Lexicon.tsv")
    vad_df = pd.read_csv(vad_file, index_col=0, sep="\t")

    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

    data_dir = pjoin(os.getcwd(), os.pardir, "data")
    agency_power_lexicon_file = pjoin(data_dir, "FramesAgencyPower", "agency_power.csv")
    ap_df = pd.read_csv(agency_power_lexicon_file, index_col=0)

    def sort_preds_by_counts(pred_dict, sort_value="count"):

        sorted_pred_dict = sorted(pred_dict.items(), key=lambda tup: tup[1][sort_value], reverse=True)

        return sorted_pred_dict

    tested_files = {}

    for masked_dir, template_dirs in tqdm(unique_dirs.items()):
        #     for masked_dir, template_dirs in unique_dirs.items():
        basename = os.path.basename(masked_dir)
        typ = basename.lower() if "_" not in basename else basename.split("_")[0].lower()
        pos = get_pos(typ)
        print(basename)
        print(typ)

        if typ != "pronouns":
            continue

        if typ == "verb":
            ap_df = pd.read_csv(agency_power_lexicon_file, index_col=0)

        template_stats = stats.get("pred_" + typ, {})

        tested_files[basename] = tested_files.get(basename, {})

        template_top_verbs = {}

        c = 0
        for template_dir, sentence_dirs in tqdm(template_dirs.items()):
            #         for template_dir in sorted(template_dirs.keys()):
            basename_temp = os.path.basename(template_dir)

            # only look at normal and type 'a' sentences
            #             if "_norm" not in template_dir or "_a_" not in template_dir:
            #                 continue

            #             c += 1
            #             if c > 10:
            #                 break

            sentence_dirs = template_dirs[template_dir]

            template_top_verbs["top_verbs"] = template_top_verbs.get("top_verbs", {})

            for preds_file in sorted(sentence_dirs):
                tested_files[basename][basename_temp] = tested_files[basename].get(basename_temp, []) + [preds_file]

                if not os.path.isfile(preds_file):
                    continue
                try:
                    with open(preds_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    adjective = data["adjective"]
                    occupation = data["occupation(0)"]
                    participant = data["other-participant(1)"]
                    answer = data["answer"]
                    pronoun = data["pronoun"]
                    sentence = data["masked_sentence"]
                    verb = data["verb"]
                    preds = data["mask_preds"]

                    # get pos, vad and ap results
                    if typ != "pronouns" and typ != "occupation":
                        pos_preds_file = preds_file.replace("preds.json", "pos_vad_preds.json")

                        if typ == "verb":
                            pos_preds = get_pos_preds(preds, pos, vad_df, ap_df, lemmatizer)
                            pos_preds_file = preds_file.replace("preds.json", "pos_vad_ap_preds.json")
                        else:
                            pos_preds = get_pos_preds(preds, pos, vad_df)

                        # filter top k predictions
                        if top_k:
                            pos_preds = pos_preds[:top_k]

                    #                         pos_pred_data = {"pos_preds": pos_preds}
                    #
                    #                         # save preds
                    #                         with open(pos_preds_file, "w", encoding="utf-8") as f:
                    #                             json.dump(pos_pred_data, f, ensure_ascii=False, indent=4)
                    else:
                        if top_k:
                            pos_preds = convert_preds(preds, typ=typ)[:top_k]
                        else:
                            pos_preds = convert_preds(preds, typ=typ)

                    opp = True if "_norm" in preds_file else False
                    someone = True if "_someone" in preds_file else False

                    template_stats = get_template_stats(template_stats, pos_preds, typ, adjective, verb, occupation,
                                                        participant, answer, pronoun, someone, opp)

                    if typ == "verb":
                        template_top_verbs["top_verbs"] = add_top_verbs(pos_preds, template_top_verbs["top_verbs"])


                except Exception:
                    traceback.print_exc()
        #             break  # template dirs
        stats["pred_" + typ] = template_stats

    #         break  # masked dirs

    final_stats = {}
    final_stats["stats"] = stats
    final_stats["tested_files"] = tested_files
    final_stats["template_top_verbs"] = template_top_verbs

    # save statistics
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=4)

    # return final_stats


def main():
    save_dir = pjoin(os.getcwd(), os.pardir, "experiments", "winogender", "BERT", "masked_token_prediction")
    unique_dirs_file = pjoin(save_dir, "unique_sentences.json")

    with open(unique_dirs_file, "r", encoding="utf-8") as f:
        unique_dirs = json.load(f)

    save_path = pjoin(save_dir, "unique_sentences_mlm_pred_pronoun_stats.json")

    top_k = 100

    # get and save stats
    generate_vad_ap_preds_and_stats(unique_dirs, save_path, top_k)


if __name__ == "__main__":
    main()