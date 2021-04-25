import os
import re
import time
import datetime
import logging
import traceback
import sys
import json
import operator
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
# set abbreviated functions
pjoin = os.path.join
listdir = os.listdir
isfile = os.path.isfile
isdir = os.path.isdir


def mkdir(path):
    if not isdir(path):
        os.mkdir(path)


class Heatmap:
    def __init__(self, results_dir, output_dir, top_k):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.top_k = top_k
        self.typ = None
        self.someone = None
        self.vad_df = None

    def generate_heatmaps(self):
        masked_item_dirs = sorted([pjoin(self.results_dir, d) for d in listdir(self.results_dir)])

        for masked_item_dir in masked_item_dirs:
            basename = os.path.basename(masked_item_dir)
            output_masked_item_dir = pjoin(self.output_dir, basename)
            mkdir(output_masked_item_dir)

            if "_" in basename:
                self.typ = basename.split("_")[0].lower()
                self.someone = True
            else:
                self.typ = basename.lower()
                self.someone = False

            # load valence-arousal-dominance lexicon
            if self.typ == "verb":
                vad_file = pjoin(os.getcwd(), os.pardir, os.pardir, "data", "NRC-VAD-Lexicon", "NRC-VAD-Lexicon.tsv")
                self.vad_df = pd.read_csv(vad_file, index_col=0, sep="\t")

            template_dirs = sorted([pjoin(masked_item_dir, f) for f in listdir(masked_item_dir)])

            for template_dir in template_dirs:
                output_template_dir = pjoin(output_masked_item_dir, os.path.basename(template_dir))
                # print(template_dir)

                # TODO if norm or opp
                sentence_dirs = sorted([pjoin(template_dir, f) for f in listdir(template_dir)])

                output = self.get_predictions_for_genders(sentence_dirs)
                print("\n", self.typ, basename)
                # print(output)

                if output == {}:
                    continue

                # x_words, y_words, weights, title = self.get_heatmap_data(output)
                # print(x_words, y_words, weights, title)
                if self.typ == "adjectives":
                    save_path = output_template_dir.replace(".tsv", ".png")
                    # self.plot_heatmap(x_words, y_words, weights, title, save_path=save_path)
                    x_words, y_words, weights, labels, title = self.get_heatmap_data_3(output)
                    self.plot_heatmap_2(y_words, weights, labels, title, save_path=save_path)
                    # print("Saved to", save_path)

                elif self.typ in ["occupation", "participant", "verb"]:
                    save_dir = output_template_dir.replace(".tsv", "")
                    mkdir(save_dir)
                    m_save_path, f_save_path, n_save_path = pjoin(save_dir, "male.png"), pjoin(save_dir, "female.png"), pjoin(save_dir, "neutral.png")
                    m_save_path_vad, f_save_path_vad, n_save_path_vad = pjoin(save_dir, "male_vad.png"), pjoin(save_dir, "female_vad.png"), pjoin(save_dir, "neutral_vad.png")

                    # x_words, y_words, weights, labels, title = self.get_heatmap_data_3(output)
                    if self.typ == "verb":
                        m_y_words, f_y_words, n_y_words, m_weights, f_weights, n_weights, m_labels, f_labels, n_labels, m_vad, f_vad, n_vad, title = self.get_heatmap_data_3(output)
                    else:
                        m_y_words, f_y_words, n_y_words, m_weights, f_weights, n_weights, m_labels, f_labels, n_labels, title = self.get_heatmap_data_3(output)

                    m_title, f_title, n_title = title

                    try:
                        self.plot_heatmap_2(m_y_words, m_weights, m_labels, m_title, save_path=m_save_path)
                        self.plot_heatmap_2(f_y_words, f_weights, f_labels, f_title, save_path=f_save_path)
                        self.plot_heatmap_2(n_y_words, n_weights, n_labels, n_title, save_path=n_save_path)

                        if self.typ == "verb":
                            self.plot_heatmap_2(m_y_words, m_weights, m_vad, m_title, save_path=m_save_path_vad)
                            self.plot_heatmap_2(f_y_words, f_weights, f_vad, f_title, save_path=f_save_path_vad)
                            self.plot_heatmap_2(n_y_words, n_weights, n_vad, n_title, save_path=n_save_path_vad)
                    except IndexError:
                        print("Couldn't plot")
                        traceback.print_exc()

                elif self.typ == "pronouns":
                    save_path = output_template_dir.replace(".tsv", ".png")
                    x_words, y_words, weights, title = self.get_heatmap_data_3(output)
                    self.plot_heatmap(x_words, y_words, weights, title, save_path=save_path)
                break
            # break

    def plot_heatmap_2(self, y_words, weights, labels, title, save_path=None, display=False):
        try:
            # if self.typ == "adjectives":
            n_rows = len(y_words)
            # print(labels)
            n_cols = len(labels[0])
            fig = plt.figure(figsize=(n_cols*1.5, n_rows*2))

            ax = sn.heatmap(weights, annot=labels, xticklabels=False, yticklabels=y_words,
                            cbar_kws={'label': "Prediction probability (in %)",
                                      "orientation": "vertical"},
                            fmt="", cmap="YlGnBu", square=False, annot_kws={"size": 12})
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.yticks(rotation=0)
            plt.title(title)

            if save_path:
                fig.savefig(save_path, dpi=300)
                print("Saving figure.")
            if not display:
                plt.close(fig)
        except IndexError:
            pass

    def plot_heatmap(self, x_words, y_words, weights, title, save_path=None, display=False):
        try:
            # if self.typ == "adjectives":
            n_rows = len(y_words) if len(y_words) > 1 else 4
            n_cols = len(x_words) if len(x_words) > 1 else 10
            fig = plt.figure(figsize=(n_cols*2, n_rows*1.5))
            sn.set(font_scale=1.5)
            ax = sn.heatmap(weights, xticklabels=x_words, yticklabels=y_words,
                            cbar_kws={'label': "Prediction probability (in %)",
                                      "orientation": "vertical"},
                            annot=True, square=False, cmap="YlGnBu",
                            annot_kws={"size":20})
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.yticks(rotation=0)
            plt.title(title)

            if save_path:
                fig.savefig(save_path, dpi=300)
                print("Saving figure.")
            if not display:
                plt.close(fig)
        except IndexError:
            pass

    def get_heatmap_data_3(self, data):
        y_words = []
        x_words = []
        heatmap_weights = []
        labels = []

        if self.typ == "adjectives":
            for k, v in data.items():
                sentence = data[k]["masked_sentence"]
                pronoun = data[k]["pronoun"]

                words, weights = zip(*[(tup[1][0], tup[1][1]) for tup in data[k]["top_k_preds"]])
                l = ["{}\n{}".format(tup[1][0], str(tup[1][1])) for tup in data[k]["top_k_preds"]]

                y_words.append(pronoun)
                x_words.append(list(words))
                heatmap_weights.append(weights)
                labels.append(l)

            title = self.generate_heatmap_title(sentence, pronoun, "PRONOUN")

            return np.array(x_words), np.array(y_words), np.array(heatmap_weights), np.array(labels), title

        elif self.typ in ["occupation", "participant", "verb"]:

            m_y_words = []
            f_y_words = []
            n_y_words = []
            m_x_words = []
            f_x_words = []
            n_x_words = []
            m_heatmap_weights = []
            f_heatmap_weights = []
            n_heatmap_weights = []
            m_labels = []
            f_labels = []
            n_labels = []
            m_vad_scores = []
            f_vad_scores = []
            n_vad_scores = []
            title = ["", "", ""]

            for k, v in data.items():
                sentence = data[k]["masked_sentence"]
                adjective = data[k]["adjective"]
                pronoun = data[k]["pronoun"]

                words, weights = zip(*[(tup[1][0], tup[1][1]) for tup in data[k]["top_k_preds"]])
                l = ["{}\n{}".format(tup[1][0], str(tup[1][1])) for tup in data[k]["top_k_preds"]]
                if self.typ == "verb":
                    vad_score = ["{d[3]}\nV:{d[0]}\nA:{d[1]}\nD:{d[2]}".format(d=self.get_vad_scores(w) + [w]) for w in words]

                # print(weights)
                # weights, l = self.pad_weights_labels(weights, l)  # pad to shape

                if pronoun in ["he", "his", "him"]:
                    m_y_words.append(adjective)
                    m_x_words.append(words)
                    m_heatmap_weights.append(weights)
                    m_labels.append(l)
                    title[0] = sentence
                    if self.typ == "verb":
                        m_vad_scores.append(vad_score)
                        title[0] = self.generate_heatmap_title(sentence, adjective.split("_")[-1], "ADJECTIVE") if adjective != "nan" else sentence
                elif pronoun in ["she", "her"]:
                    f_y_words.append(adjective)
                    f_x_words.append(words)
                    f_heatmap_weights.append(weights)
                    f_labels.append(l)
                    title[1] = sentence
                    if self.typ == "verb":
                        f_vad_scores.append(vad_score)
                        title[1] = self.generate_heatmap_title(sentence, adjective.split("_")[-1], "ADJECTIVE") if adjective != "nan" else sentence
                elif pronoun in ["they", "their", "them"]:
                    n_y_words.append(adjective)
                    n_x_words.append(words)
                    n_heatmap_weights.append(weights)
                    n_labels.append(l)
                    title[2] = sentence
                    if self.typ == "verb":
                        n_vad_scores.append(vad_score)
                        title[2] = self.generate_heatmap_title(sentence, adjective.split("_")[-1], "ADJECTIVE") if adjective != "nan" else sentence

            m_heatmap_weights, m_labels = self.pad_weights_labels(m_heatmap_weights, m_labels)
            f_heatmap_weights, f_labels = self.pad_weights_labels(f_heatmap_weights, f_labels)
            n_heatmap_weights, n_labels = self.pad_weights_labels(n_heatmap_weights, n_labels)
            if self.typ == "verb":
                m_vad_scores = np.array([list(r) + (self.top_k - len(r)) * [""] for r in m_vad_scores], dtype=np.str)
                f_vad_scores = np.array([list(r) + (self.top_k - len(r)) * [""] for r in f_vad_scores], dtype=np.str)
                n_vad_scores = np.array([list(r) + (self.top_k - len(r)) * [""] for r in n_vad_scores], dtype=np.str)
                # m_title = self.generate_heatmap_title(title[1], adjective.split("_")[-1], "ADJECTIVE")
                # f_title = self.generate_heatmap_title(title[0], adjective.split("_")[-1], "ADJECTIVE")
                # n_title = self.generate_heatmap_title(title[2], adjective.split("_")[-1], "ADJECTIVE")
                # title = [m_title, f_title, n_title]

                return np.array(m_y_words), np.array(f_y_words), np.array(n_y_words), np.array(m_heatmap_weights), np.array(f_heatmap_weights), np.array(n_heatmap_weights), np.array(m_labels), np.array(f_labels), np.array(n_labels), np.array(m_vad_scores), np.array(f_vad_scores), np.array(n_vad_scores), title

            return np.array(m_y_words), np.array(f_y_words), np.array(n_y_words), np.array(m_heatmap_weights), np.array(f_heatmap_weights), np.array(n_heatmap_weights), np.array(m_labels), np.array(f_labels), np.array(n_labels), title

        elif self.typ == "pronouns":
            for k,v in data.items():
                sentence = data[k]["masked_sentence"]
                adjective = data[k]["adjective"]

                words, weights = zip(*[(v1[0], v1[2]) for k1,v1 in data[k]["top_k_preds"].items()])
                words, weights = self.sort_pronouns(words, weights)

                y_words = words
                x_words.append(adjective)
                heatmap_weights.append(weights)

                if adjective != "nan":
                    title = self.generate_heatmap_title(sentence, adjective.split("_")[-1], "ADJECTIVE")
                else:
                    title = sentence

            x_words, heatmap_weights = self.sort_adjectives(x_words, heatmap_weights)

            return np.array(x_words), np.array(y_words), np.array(heatmap_weights).T, title

    def get_vad_scores(self, word):
        if word in self.vad_df.index:
            return [str(round(v, 4)) for v in self.vad_df.loc[word]]
        else:
            return ["", "", ""]  # TODO make sure both returns are same type

    def sort_pronouns(self, words, weights):
        assert (len(words) == len(weights))
        pronouns = ["she", "her", "he", "his", "him", "they", "their", "them"]

        new_words = []
        new_weights = []

        for p in pronouns:
            for i, v in enumerate(words):
                if v == p:
                    new_words.append(v)
                    new_weights.append(weights[i])

        return new_words, new_weights

    def sort_adjectives(self, words, weights):
        sorted_words = sorted(words)

        new_weights = []

        for word in sorted_words:
            for i, w in enumerate(words):
                if w == word:
                    new_weights.append(weights[i])

        return sorted_words, new_weights

    def get_heatmap_data_2(self, data):
        y_words = []
        x_words = []
        heatmap_weights = []
        labels = []
        title = ""

        if self.typ == "adjectives":
            for k, v in data.items():
                sentence = data[k]["masked_sentence"]
                pronoun = data[k]["pronoun"]

                words, weights = zip(*[(tup[1][0], tup[1][1]) for tup in data[k]["top_k_preds"]])
                l = ["{}\n{}".format(tup[1][0], str(tup[1][1])) for tup in data[k]["top_k_preds"]]

                y_words.append(pronoun)
                x_words.append(list(words))
                heatmap_weights.append(weights)
                labels.append(l)

                title = self.generate_heatmap_title(sentence, pronoun)

        return np.asarray(x_words), np.asarray(y_words), np.asarray(heatmap_weights), np.asarray(labels), title

    def get_heatmap_data(self, data):
        y_words = []
        x_words = []
        heatmap_weights = []
        title = ""

        if self.typ == "adjectives":

            for k,v in data.items():
                sentence = data[k]["masked_sentence"]
                pronoun = data[k]["pronoun"]

                words, weights = zip(*[(tup[1][0], tup[1][1]) for tup in data[k]["top_k_preds"]])

                y_words.append(pronoun)
                x_words.append(list(words))
                heatmap_weights.append(weights)

                title = self.generate_heatmap_title(sentence, pronoun, "PRONOUN")

        return x_words, y_words, heatmap_weights, title

    def pad_weights_labels(self, weights, labels):
        weights = self.pad_to_dense(np.array(weights))
        labels = np.array([list(r) + (self.top_k - len(r)) * [""] for r in labels], dtype=np.str)

        return weights, labels

    def pad_to_dense(self, arr, typ=np.float):

        z = np.zeros((arr.shape[0], self.top_k), dtype=typ)
        for enu, row in enumerate(arr):
            z[enu, :len(row)] += row
        return z

    def generate_heatmap_title(self, sentence, pronoun, placeholder):
        toks = sentence.split(" ")

        word_idx = toks.index(pronoun)
        new_toks = toks[:word_idx] + ["<"+placeholder+">"] + toks[word_idx+1:]

        return " ".join(new_toks)

    def get_top_k_preds_with_pos(self, preds, pos=None):
        pos_preds = {}

        if pos != "pronoun" and pos != None: # if noun, adjective or verb
            for k,v in preds.items():
                if wn.morphy(v[0], pos):
                    pos_preds[int(k)] = v  # convert string key to int key for sorting
        elif pos == "pronoun":
            pos_preds = preds
        else:
            pos_preds = preds

        top_k_preds = sorted(pos_preds.items(), key=lambda kv: kv[0])

        return top_k_preds[:self.top_k]

    def get_predictions_for_genders(self, sentence_dirs):
        output = {}
        for sentence_dir in sentence_dirs:
            # if preds file exists generate heatmap, else skip
            preds_file = pjoin(sentence_dir, "preds.json")
            if isfile(pjoin(preds_file)):
                # read json
                with open(preds_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                masked_sent = data["masked_sentence"]
                occ = data["occupation(0)"]
                par = data["other-participant(1)"]
                answer = data["answer"]
                adjective = data["adjective"]
                verb = data["verb"]
                pronoun = data["pronoun"]

                mask_preds = data["mask_preds"]

                if self.typ in ["occupation", "participant"]:
                    pos = wn.NOUN
                elif self.typ == "adjectives":
                    pos = wn.ADJ
                elif self.typ == "verb":
                    pos = wn.VERB
                elif self.typ == "pronouns":
                    pos = "pronoun"
                else:
                    pos = ""
                    continue

                if self.typ == "pronouns":
                    top_k_preds = mask_preds
                elif self.typ not in ["occupation", "participant"]:
                    top_k_preds = self.get_top_k_preds_with_pos(mask_preds, pos)
                else:
                    top_k_preds = self.get_top_k_preds_with_pos(mask_preds)

                _ = {}
                _["masked_sentence"] = masked_sent
                _["type"] = self.typ
                _["top_k_preds"] = top_k_preds
                _["adjective"] = adjective
                _["occupation(0)"] = occ
                _["other-participant(1)"] = par
                _["answer"] = answer
                _["verb"] = verb
                if self.typ != "pronouns":
                    if self.someone and pronoun in ["they", "their", "them"]:
                        _["pronoun"] = pronoun
                    elif pronoun not in ["they", "their", "them"]:
                        _["pronoun"] = pronoun
                    else:
                        continue

                output[os.path.basename(sentence_dir)] = _
            else:
                continue

        return output


if __name__ == "__main__":
    # log process
    log_dir = pjoin(os.getcwd(), ".logs")
    mkdir(log_dir)
    logging.basicConfig(filename=pjoin(log_dir, "masked_token_heatmaps_" + "-".join(
        re.split(":|\.| ", str(datetime.datetime.now()))[:-1]) + ".log"),
                        level=logging.INFO)

    logging.info("Started.")

    root_dir = pjoin(os.getcwd(), os.pardir, os.pardir)
    experiments_dir = pjoin(root_dir, "experiments", "winogender", "BERT", "masked_token_prediction")
    analysis_dir = pjoin(root_dir, "analysis", "winogender", "BERT", "masked_token_prediction")

    models = ["bert-large-cased"]
    top_k = 20

    for model in models:
        results_dir = pjoin(experiments_dir, model)
        output_dir = pjoin(analysis_dir, model)
        mkdir(output_dir)

        h = Heatmap(results_dir, output_dir, top_k)

        h.generate_heatmaps()
