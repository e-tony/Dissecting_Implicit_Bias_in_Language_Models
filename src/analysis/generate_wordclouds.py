from nltk.corpus import wordnet as wn
import seaborn as sn
import matplotlib.pyplot as plt
import os, json
import pandas as pd
import json
import traceback
from PIL import Image
from wordcloud import WordCloud
from tqdm import tqdm
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
# lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
pjoin = os.path.join


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def sort_preds_by_x(pred_dict, sort_value="count"):
    sorted_pred_dict = sorted(pred_dict.items(), key=lambda tup: tup[1][sort_value], reverse=True)

    return sorted_pred_dict


def get_avg_pred_perc(preds, typ="pronouns"):
    percentages = {}

    if typ == "pronouns":
        percentages = {w: round(sum(d["percent"]) / len(d["percent"]), 3) for w, d in preds.items() if
                       len(w) > 1 and not any(i.isdigit() for i in w)}

    return percentages


def convert_noun_vad_preds(preds):
    output = {}

    for word, d in preds.items():
        _d = {}
        if "ap" in d.keys():
            _d["agency"] = d["ap"]["a"]
            _d["power"] = d["ap"]["p"]
        _d["percent"] = d["percent"]
        _d["valence"] = d["vad"]["valence"]
        _d["arousal"] = d["vad"]["arousal"]
        _d["dominance"] = d["vad"]["dominance"]
        _d["count"] = d["count"]

        output[word] = _d

    return output


def plot_pronoun_percentages(preds, get_percentages=False, save_path=None):
    avg_perc = get_avg_pred_perc(preds)

    _df = pd.DataFrame(list(avg_perc.items()), columns=["pronoun", "percent"])

    # plt.figure(figsize=(10,10))
    ax = _df.plot(kind="bar", x="pronoun", y="percent", rot=0, colormap="Accent")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Prediction percentage")
    ax.set_xlabel("")
    ax.set_title("Pronoun Prediction Distribution")

    for i in range(_df.shape[0]):
        pronoun, percent = _df.loc[i]
        percent = round(percent, 1)
        ax.annotate(str(percent), (i - 0.26, percent + 2))

    if get_percentages:
        return avg_perc
    if save_path:
        fig = ax.get_figure()
        fig.savefig(save_path)


def generate_wordcloud(pred_dict, flip_weights=False, typ="adjectives", value="count", save_path=None, show=None):
    def get_avg(weights, value, save_path):
        avg = round(sum([v for k, v in weights.items()]) / len(weights), 3)
        #         title = "Average '{}': {}".format(value, str(avg))
        save_path = save_path.replace(".png", "-avg={}.png".format(str(avg)))
        return avg, save_path

    try:
        title = None

        if typ in ["adjectives", "verbs"] and value in ["valence", "arousal", "dominance"]:
            print("In 1")
            _weights = {w: d["vad"][value] for w, d in sort_preds_by_x(pred_dict) if
                        len(w) > 1 and not any(i.isdigit() for i in w) and d["vad"][value] != 0.0}
            if save_path:
                avg, save_path = get_avg(_weights, value, save_path)
        elif value == "percent":
            print("In 2")
            _weights = {w: sum(d["percent"]) / len(d["percent"]) for w, d in sort_preds_by_x(pred_dict) if
                        len(w) > 1 and not any(i.isdigit() for i in w) and d["percent"] != 0.0}
            if save_path:
                avg, save_path = get_avg(_weights, value, save_path)
        else:
            print("In 3")
            _weights = {w: d[value] for w, d in sort_preds_by_x(pred_dict) if
                        len(w) > 1 and not any(i.isdigit() for i in w)}

        if flip_weights and value in ["valence", "arousal", "dominance"]:
            _weights = {w: round(1.0 - v, 3) for w, v in _weights.items() if v != 0.0}
        else:
            _weights = {w: round(v, 3) for w, v in _weights.items() if v != 0.0}

        # plot wordcloud
        wc = WordCloud(max_font_size=50, background_color="white")
        wordcloud = wc.generate_from_frequencies(_weights)

        if show:
            plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        if save_path:
            wc.to_file(save_path)
    #             plt.savefig(save_path)
    except Exception:
        traceback.print_exc()
        # print("FAILED for file:", save_path)
        # print(pred_dict)


def generate_wordclouds(final_stats, wc_save_dir):
    wc_save_dir = pjoin(os.getcwd(), os.pardir, os.pardir, "images", "wordclouds_3")

    # for pred_task in ["pred_verb", "pred_adjectives", "pred_pronouns"]:
    for pred_task in ["pred_verb", "pred_adjectives", "pred_pronouns"]:
        for someone in ["not_someone", "someone"]:  # 'someone' doesnt exist for adjectives
            if final_stats["stats"][pred_task][someone] == {}:  # skip empty 'someone' dicts
                continue

            if pred_task == "pred_verb":
                pred_types = ["adjectives", "verbs", "occupations", "participants"]
            elif pred_task == "pred_adjectives":
                pred_types = ["adjectives"]
            elif pred_task == "pred_pronouns":
                pred_types = ["pronouns"]

            for pred_type in pred_types:
                _ = final_stats["stats"][pred_task][someone][pred_type]
                # print(_.keys())
                preds = _
                if pred_type in ["adjectives", "verbs"]:

                    if preds == {}:
                        continue
                    for wc_type in ["valence", "arousal", "dominance", "count", "percent"]:
                        if wc_type in ["valence", "arousal", "dominance"]:
                            for flip in [False, True]:
                                _save_dir = pjoin(wc_save_dir, pred_task, pred_type)
                                mkdir(_save_dir)
                                save_path = pjoin(_save_dir, "-".join([pred_task, someone, pred_type, wc_type, "flip="+str(flip)])+".png")
                                generate_wordcloud(preds, flip_weights=flip, typ=pred_type, value=wc_type, save_path=save_path)
                        else:
                            _save_dir = pjoin(wc_save_dir, pred_task, pred_type)
                            mkdir(_save_dir)
                            save_path = pjoin(_save_dir, "-".join([pred_task, someone, pred_type, wc_type])+".png")
                            generate_wordcloud(preds, typ=pred_type, value=wc_type, save_path=save_path)

                elif pred_type in ["occupations","participants"]:
                    continue
                    for noun, preds in _.items():
                        if preds == {}:
                            continue

                        # convert nouns preds to correct format
                        preds = convert_noun_vad_preds(preds)
                        for wc_type in ["valence", "arousal", "dominance", "count", "percent"]:
                            _save_dir = pjoin(wc_save_dir, pred_task, pred_type)
                            mkdir(_save_dir)
                            if wc_type in ["valence", "arousal", "dominance"]:
                                for flip in [False, True]:
                                    save_path = pjoin(_save_dir, "-".join(
                                        [pred_task, someone, pred_type, noun, wc_type, "flip=" + str(flip)]) + ".png")
                                    generate_wordcloud(preds, flip_weights=flip, typ=pred_type, value=wc_type,
                                                       save_path=save_path)
                            else:
                                save_path = pjoin(_save_dir,
                                                  "-".join([pred_task, someone, pred_type, noun, wc_type]) + ".png")
                                generate_wordcloud(preds, typ=pred_type, value=wc_type, save_path=save_path)

                elif pred_type == "pronouns":
                    preds = _
                    wc_type = "percent"
                    _save_dir = pjoin(wc_save_dir, pred_task)
                    mkdir(_save_dir)
                    save_path = pjoin(_save_dir, "-".join([pred_task, someone, pred_type, wc_type + ".png"]))
                    plot_pronoun_percentages(preds, get_percentages=False)

                else:
                    continue


def main():
    experiments_dir = pjoin(os.getcwd(), os.pardir, os.pardir, "experiments", "winogender", "BERT", "masked_token_prediction")
    final_stats_file = pjoin(experiments_dir, "unique_sentences_mlm_pred_stats.json")

    with open(final_stats_file, "r", encoding="utf-8") as f:
        final_stats = json.load(f)

    wc_save_dir = pjoin(os.getcwd(), os.pardir, "images", "wordclouds")
    mkdir(wc_save_dir)

    generate_wordclouds(final_stats, wc_save_dir)


if __name__ == "__main__":
    main()