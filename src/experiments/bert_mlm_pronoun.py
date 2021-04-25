import torch
from pytorch_transformers import (BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction,
                                  BertForTokenClassification, BertForMultipleChoice)
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


def predict_masked_token(model, tokenizer, mask_path, experiment_path, model_name, top_k=10):
    experiments_dir_bert = pjoin(experiment_path, "winogender", "BERT", "masked_token_prediction")
    mkdir(experiments_dir_bert)

    sentence_dirs = sorted([pjoin(mask_path, d) for d in listdir(mask_path)])

    _template = "occ-{}_par-{}_answer-{}_adj-{}_verb-{}_pronoun-{}_masksent-{}"

    # create save dir
    model_dir = pjoin(experiments_dir_bert, model_name)
    mkdir(model_dir)

    for sentence_dir in tqdm(sentence_dirs):
        # check if predicting pronouns
        masked_pronoun = False
        if "pronoun" in sentence_dir:
            masked_pronoun = True

        print("Processing sentence type: {}...".format(os.path.basename(sentence_dir)))
        logging.info("Processing '{}' sentence directory.".format(sentence_dir))

        # create sentence type save dir
        type_dir = pjoin(model_dir, os.path.basename(sentence_dir))
        mkdir(type_dir)

        files = sorted([pjoin(sentence_dir, f) for f in listdir(sentence_dir)])

        for file in tqdm(files):
            logging.info("Processing file: '{}'.".format(file))

            # read file
            df = pd.read_csv(file, sep="\t")
            columns = list(df.columns)
            n_rows = df.shape[0]  # number of rows

            if n_rows < 1:  # skip empty files
                logging.info("Skipping emtyp file: '{}'.".format(file))
                continue

            # create file save dir
            file_dir = pjoin(type_dir, os.path.basename(file))
            mkdir(file_dir)

            # list of sentences already processed
            seen = []

            for i in range(n_rows):
                # weights_path = pjoin(name_dir, "preds.json")
                # name_dir = pjoin(file_dir, str(i))
                #
                # # skip if file already exists
                # if isfile(weights_path):
                #     logging.info("'preds.json' file already exists for row: {}. Skipping...".format(i))
                #     continue

                try:
                    logging.info("Processing row '{}'.".format(i))

                    row = df.iloc[i]
                    occ, par, answer, adj, verb, pronoun, mask_sent, _sent = row

                    #                 print(mask_sent)
                    if mask_sent in seen:
                        logging.info("Sentence in row '{}' is a duplicate and has already been processed in this run.".format(i))
                        continue
                    else:
                        seen.append(mask_sent)

                    # create sentence prediction results dir
                    # name = _template.format(occ, par, answer, adj, "+".join(verb.split(" ")),
                    #                         pronoun, "+".join(
                    #         mask_sent[:-1].replace("[", "").replace("]", "").split(" ") + ["."]))
                    # name_dir = pjoin(file_dir, name)
                    name_dir = pjoin(file_dir, str(i+1))
                    mkdir(name_dir)
                    weights_path = pjoin(name_dir, "preds.json")

                    # skip if 'preds.json' file already exists
                    if isfile(weights_path):
                        logging.info("'preds.json' file already exists for row: {}. Skipping...".format(i))
                        continue

                    # tokenize sentence
                    tokenized_sent = tokenizer.tokenize(mask_sent)
                    mask_idx = tokenized_sent.index("[MASK]")

                    # convert tokens to indices of vocabulary
                    indexed_toks = tokenizer.convert_tokens_to_ids(tokenized_sent)

                    # Split into 2 sentences for BERT task
                    sent_1 = mask_idx
                    sent_2 = len(tokenized_sent) - mask_idx
                    segment_ids = [0] * sent_1 + [1] * sent_2

                    # Convert inputs to PyTorch tensors
                    token_tensors = torch.tensor([indexed_toks])
                    segment_tensors = torch.tensor([segment_ids])

                    # Predict masked token
                    outputs = model(token_tensors, segment_tensors)
                    logits = outputs[0]
                    # logits, states, attentions = outputs
                    # # save attentions & states
                    # torch.save(states, pjoin(name_dir "states.pt"))
                    # torch.save(attentions, pjoin(name_dir, "attentions.pt"))

                    if masked_pronoun:
                        # possible candidate words to predict
                        candidates = ["she", "her", "he", "his", "him", "they", "their", "them"]
                        candidates_ids = tokenizer.convert_tokens_to_ids(candidates)
                        prediction_candidates = logits[0, mask_idx, candidates_ids]

                        probs = torch.nn.functional.softmax(prediction_candidates, dim=-1)
                        sorted_probs = sorted(probs)

                        preds = {}

                        for j, prob in enumerate(probs):
                            for i, s_prob in enumerate(sorted_probs):
                                if s_prob.item() == prob.item():
                                    preds[i + 1] = [candidates[j], prediction_candidates[j].item(),
                                                    round(s_prob.item() * 100, 3)]
                    else:
                        probs = torch.nn.functional.softmax(logits[0, mask_idx], dim=-1)

                        # get top k predicted weights, tokens indices, & probabilities
                        top_weights, top_idx = torch.topk(logits[0][mask_idx], top_k, sorted=True)
                        top_probs, top_idx = torch.topk(probs, top_k, sorted=True)

                        # write weights & tokens in lists
                        pred_weights = []
                        pred_tokens = []

                        for k in range(top_k):
                            # convert idx to token
                            _pred_idx = top_idx[k].item()
                            _pred_tok = tokenizer.convert_ids_to_tokens([_pred_idx])[0]

                            # add to lists
                            pred_tokens.append(_pred_tok)
                            pred_weights.append(round(top_weights[k].item(), 1))  # round to 1 decimal position

                        assert (len(pred_weights) == len(pred_tokens))

                        # prediction token & weight mapping
                        preds = {}

                        # key = index in list, value = (token, weight) tuple
                        for j, tok in enumerate(pred_tokens):
                            preds[str(j + 1)] = (tok, round(float(top_probs[j] * 100), 2), float(pred_weights[j]))

                        # map predictions & column values into final json
                    output_json = {}
                    output_json["mask_preds"] = preds

                    for col in columns:
                        # convert to proper type
                        if type(row[col]) == np.int64:
                            output_json[col] = int(row[col])
                        elif row[col] == "nan":
                            output_json[col] = str(row[col])
                        else:
                            output_json[col] = str(row[col])

                    # create weights json file
                    weights_path = pjoin(name_dir, "preds.json")
                    write_json(output_json, weights_path)

                    logging.info("Done processing row '{}'.".format(i))
                    # break
                except Exception as e:
                    logging.info("ERROR: During processing row '{}' an error occured.".format(i))
                    logging.info(traceback.format_exc())
                    logging.info(sys.exc_info()[0])

            logging.info("Done processing file: '{}'.".format(file))
            # break
        logging.info("Done processing sentences directory '{}'.".format(sentence_dir))
        # break


if __name__ == "__main__":
    # log process
    log_dir = pjoin(os.getcwd(), ".logs")
    mkdir(log_dir)
    logging.basicConfig(filename=pjoin(log_dir, "masked_token_"+"-".join(re.split(":|\.| ", str(datetime.datetime.now()))[:-1])+".log"),
                        level=logging.INFO)

    logging.info("Started.")

    winogender_masked_dir = pjoin(os.getcwd(), os.pardir, os.pardir, "data", "winogender-adj-verb", "masked_v2")
    experiment_dir = pjoin(os.getcwd(), os.pardir, os.pardir, "experiments")

    top_k = 1000
    # bert_models = ['bert-large-cased', 'bert-large-uncased', 'bert-base-cased', 'bert-base-uncased']
    bert_models = ['bert-large-cased']

    for model_name in bert_models:
        try:
            print("Processing model: '{}'...".format(model_name))
            logging.info("Loading '{}' model...".format(model_name))

            model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=False, output_attentions=False)
            model.eval()
            tokenizer = BertTokenizer.from_pretrained(model_name)

            logging.info("Loaded '{}' model.".format(model_name))

            predict_masked_token(model, tokenizer, winogender_masked_dir, experiment_dir, model_name, top_k)

            logging.info("Done processing '{}' model.".format(model_name))
            # break
        except Exception as e:
            logging.info("ERROR: During processing model '{}' an error occured.".format(model_name))
            logging.info(traceback.format_exc())
            logging.info(sys.exc_info()[0])

    logging.info("Finished.")
