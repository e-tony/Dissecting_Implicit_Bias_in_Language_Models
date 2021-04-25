import torch
from pytorch_transformers import (BertTokenizer, BertModel, BertForMultipleChoice)
import pandas as pd
import numpy as np
import json
import os
import re
import traceback
import sys
from tqdm import tqdm
import logging
import datetime
# set abbreviated functions
pjoin = os.path.join
listdir = os.listdir
isfile = os.path.isfile
isdir = os.path.isdir


def mkdir(path):
    if not isdir(path):
        os.mkdir(path)


def write_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(file):

    return


def predict_multiple_choice(model, tokenizer, templates_path, experiment_path, model_name):
    experiments_dir_bert = pjoin(experiment_path, "winogender", "BERT", "multiple_choice_prediction")
    mkdir(experiments_dir_bert)

    templates = sorted([pjoin(templates_path, d) for d in listdir(templates_path)])
    print(templates_path)

    # create save dir
    model_dir = pjoin(experiments_dir_bert, model_name)
    mkdir(model_dir)

    for j,file in enumerate(tqdm(templates)):
        logging.info("Processing file '{}'.".format(file))

        # read file
        df = pd.read_csv(file, sep="\t")
        columns = list(df.columns)
        n_rows = df.shape[0]  # number of rows

        if n_rows < 1:  # skip empty files
            logging.info("Skipping empty file: '{}'.".format(file))
            continue

        # create file save dir
        file_dir = pjoin(model_dir, os.path.basename(file).replace("template_", "").split(".")[0])
        mkdir(file_dir)

        # list of sentences already processed
        seen = []

        # predictions
        output_preds = []
        preds_path = pjoin(file_dir, "preds.tsv")

        for i in range(n_rows):
            try:
                logging.info("Processing row '{}'.".format(i))

                row = df.iloc[i]
                _, occ, par, answer, someone, adj, verb, sent, f_sent, m_sent, n_sent = row

                if f_sent in seen or m_sent in seen or n_sent in seen:
                    logging.info("Sentence in row '{}' is a duplicate and has already been processed in this run.".format(i))
                    continue
                else:
                    seen.append(f_sent)
                    seen.append(m_sent)
                    seen.append(n_sent)

                # # create sentence prediction results dir
                # row_dir = pjoin(file_dir, str(i+1))
                # mkdir(row_dir)
                # preds_path = pjoin(row_dir, "preds.tsv")

                # # skip if 'preds.json' file already exists
                # if isfile(preds_path):
                #     logging.info("'preds.json' file already exists for row: {}. Skipping...".format(i))
                #     continue

                # 3 choices
                choices = [f_sent, m_sent, n_sent]
                input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # batch size 1
                labels = torch.tensor(1).unsqueeze(0)  # batch size 1

                outputs = model(input_ids, labels=labels)
                loss, logits = outputs[:2]

                probs = torch.nn.functional.softmax(logits, dim=-1)

                probs_rounded = [round(p*100, 2) for p in probs.tolist()[0]]

                new_row = [i, occ, par, answer, someone, adj, verb, ""] + probs_rounded

                output_preds.append(new_row)

                # # save predictions
                # new_df = pd.DataFrame([probs_rounded], index=None, columns=["female", "male", "neutral"])
                # new_df.to_csv(preds_path, sep="\t", index=True)

                logging.info("Done processing row '{}'.".format(i))
            except Exception as e:
                logging.info("ERROR: During processing row '{}' an error occured.".format(i))
                logging.info(traceback.format_exc())
                logging.info(sys.exc_info()[0])

            # break

        # save predictions
        new_df = pd.DataFrame(output_preds, index=None, columns=["row"]+columns[1:])
        new_df.to_csv(preds_path, sep="\t", index=False)
        logging.info("Done processing file '{}'.".format(file))
        # break


if __name__ == "__main__":
    # log process
    log_dir = pjoin(os.getcwd(), ".logs")
    mkdir(log_dir)
    logging.basicConfig(filename=pjoin(log_dir, "multiple_choice_"+"-".join(re.split(":|\.| ", str(datetime.datetime.now()))[:-1])+".log"),
                        level=logging.INFO)

    logging.info("Started.")

    templates_dir = pjoin(os.getcwd(), os.pardir, os.pardir, "data", "winogender-adj-verb", "multiple-choice")
    experiment_dir = pjoin(os.getcwd(), os.pardir, os.pardir, "experiments")

    bert_models = ['bert-large-cased', 'bert-base-cased']
    # bert_models = ['bert-base-uncased']

    for model_name in bert_models:
        try:
            print("Processing model: '{}'...".format(model_name))
            logging.info("Loading '{}' model...".format(model_name))

            model = BertForMultipleChoice.from_pretrained(model_name, output_hidden_states=False,
                                                          output_attentions=False)
            model.eval()
            tokenizer = BertTokenizer.from_pretrained(model_name)

            logging.info("Loaded '{}' model.".format(model_name))

            predict_multiple_choice(model, tokenizer, templates_dir, experiment_dir, model_name)

            logging.info("Done processing '{}' model.".format(model_name))
        except Exception as e:
            logging.info("ERROR: During processing model '{}' an error occured.".format(model_name))
            logging.info(traceback.format_exc())
            logging.info(sys.exc_info()[0])
        # break

    logging.info("Finished.")
