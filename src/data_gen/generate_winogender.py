from pathlib import Path
import sys
src_path = str(Path(__file__).resolve().parents[1])
sys.path.append(src_path)
import pandas as pd
import spacy
import logging
import datetime
from tqdm import tqdm
import os

# set abbreviated functions
pjoin = os.path.join
listdir = os.listdir
isfile = os.path.isfile
isdir = os.path.isdir


def mkdir(path):
    if not isdir(path):
        os.mkdir(path)


def remove_empty_char(toks):
    new_toks = []

    for tok in toks:
        if tok == '':
            continue
        new_toks.append(tok)

    return new_toks


def replace_colon(sentence):
    assert ";" in sentence

    front, back = sentence.split(";")
    sentence = ".".join([front, back[:2].upper() + back[2:]])

    return sentence


def replace_placeholder(row):
    occ_token, par_token, answer, sentence = row

    occ_placeholder = "$OCCUPATION"
    par_placeholder = "$PARTICIPANT"

    tokens = sentence.split(" ")

    occ_idx = tokens.index(occ_placeholder)
    par_idx = tokens.index(par_placeholder)

    new_tokens = tokens[:occ_idx] + [occ_token] + tokens[occ_idx + 1:]
    new_tokens = new_tokens[:par_idx] + [par_token] + new_tokens[par_idx + 1:]

    new_sentence = " ".join(new_tokens)

    return new_sentence


def insert_placeholder(placeholder, actor, sentence):
    tokens = sentence.split(" ")
    idx = tokens.index(actor)
    new_tokens = tokens[:idx] + [placeholder] + tokens[idx:]
    new_sentence = " ".join(new_tokens)

    return new_sentence


def get_placeholder_idx(placeholder, sentence):
    tokens = sentence.split(" ")
    idx = tokens.index(placeholder)

    return idx


def replace_verb_with_placeholder(sentence, occupation, participant, nlp):
    # Set placeholders
    occ_placeholder = "$OCCUPATION"
    par_placeholder = "$PARTICIPANT"

    # Split sentences with colons into two sentences
    doc = nlp(u'{}'.format(sentence))

    # Get placeholder index
    occ = [t.i for t in doc if str(t) == occupation][0]
    par = [t.i for t in doc if str(t) == participant][0]

    # List of verb tuples
    # The list should be ordered by index
    verb_idxs = []

    # Verbs only list
    verbs = []

    # Get the indices and types for verbs
    dash = False
    adp = False  # TODO might introduce errors
    roots = []

    for token in doc:
        if adp:
            continue

        if token.dep_ == "ROOT":
            roots.append(token.text)

        if token.dep_ == "acl" and token.pos_ == "VERB" and token.head.dep_ == "nsubj":
            roots.append(token.text)

        # Check for prepositions following verbs
        if token.pos_ == "ADP" and token.dep_ == "prep" or token.pos_ == "PART" and str(
                token.head) in verbs or token.dep_ == "dative" and token.pos_ != "NOUN" and str(token.head) in verbs:

            # Specific rules
            if token.text == "because":
                continue
            if "gave" in roots and token.text == "on":
                continue

            # If head is not root verb
            if str(token.head.dep_) == "conj":
                pass
            elif str(token.head) not in roots:
                continue

            adp_placeholder = "$" + token.pos_ + "_" + token.tag_
            verb_idxs.append((token.i, token.text, adp_placeholder))
            verbs.append(token.text)

            adp = True

        # Check if verb has a dash
        if token.text == "-":
            dash = True

        if token.pos_ == "VERB" and token.i > min(occ, par) and token.i < max(occ, par):
            # Skip open clausal complements. See:
            # https://universaldependencies.org/u/dep/xcomp.html

            if token.dep_ == "xcomp" or token.dep_ == "advcl":
                continue

            verb_placeholder = "$" + token.pos_ + "_" + token.tag_

            if dash:
                t_m2, t_m1, t_0 = doc[token.i - 2], doc[token.i - 1], doc[token.i]
                verb_idxs.append((token.i - 2, t_m2.text + t_m1.text + t_0.text, verb_placeholder))
                verbs.append(t_m2.text + t_m1.text + t_0.text)
                par = [t.i-2 for t in doc if str(t) == participant][0]
                continue

            verb_idxs.append((token.i, token.text, verb_placeholder))
            verbs.append(token.text)

    # Tokenize sentence
    tokens = sentence.split(" ")

    # Insert verb placeholders
    for idx, verb, placeholder in verb_idxs:
        tokens = tokens[:idx] + [placeholder] + tokens[idx + 1:]

    new_tokens = tokens[:occ] + [occ_placeholder] + tokens[occ + 1:]
    new_tokens = new_tokens[:par] + [par_placeholder] + new_tokens[par + 1:]

    new_sentence = " ".join(new_tokens)

    return new_sentence, verbs


def create_df_with_placeholders(df_path, save_dir=None):
    if not save_dir:
    # Create sentences directory
        save_dir = pjoin(os.path.dirname(df_path), "templates_"+str(datetime.datetime.now()).split(" ")[0])
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Load DataFrame
    df = pd.read_csv(df_path, sep='\t')

    # Load spacy
    nlp = spacy.load('en_core_web_lg')  # TODO try different models: 'en_core_web_md' or 'en_core_web_lg'

    rows = df.shape[0]  # number of rows
    placeholder_adj = "$ADJECTIVE"

    for i in range(rows):

        # Data to write to file
        output_data = []

        # Extract template information
        row = df.iloc[i]
        occupation, participant, answer, sentence = row

        if answer == 1:
            placeholder_noun = "$PARTICIPANT"
            placeholder_opp = "$OCCUPATION"
        elif answer == 0:
            placeholder_noun = "$OCCUPATION"
            placeholder_opp = "$PARTICIPANT"
        else:
            logging.info("Answer value is not 0 or 1 for sentence: {}".format(sentence))
            continue

        adjective_sentence = insert_placeholder(placeholder_adj, placeholder_noun, sentence)
        adjective_sentence_opp = insert_placeholder(placeholder_adj, placeholder_opp, sentence)

        # Replace placeholders with nouns
        _sentence = replace_placeholder(row)

        verb_sentence, verbs = replace_verb_with_placeholder(_sentence, occupation, participant, nlp)
        adjective_verb_sentence = insert_placeholder(placeholder_adj, placeholder_noun, verb_sentence)
        adjective_verb_sentence_opp = insert_placeholder(placeholder_adj, placeholder_opp, verb_sentence)

        plain_row = (occupation, participant, answer, None, None, sentence)
        adjective_row = (occupation, participant, answer, None, None, adjective_sentence)
        adjective_opp_row = (occupation, participant, answer, None, None, adjective_sentence_opp)
        verb_row = (occupation, participant, answer, None, " ".join(verbs), verb_sentence)
        adjective_verb_row = (occupation, participant, answer, None, " ".join(verbs), adjective_verb_sentence)
        adjective_opp_verb_row = (occupation, participant, answer, None, " ".join(verbs), adjective_verb_sentence_opp)

        # Create output DataFrame
        output_data.append(plain_row)
        output_data.append(adjective_row)
        output_data.append(adjective_opp_row)
        output_data.append(verb_row)
        output_data.append(adjective_verb_row)
        output_data.append(adjective_opp_verb_row)

        columns = list(df)[:-1] + ["adjective", "verb"] + list(df)[-1:]
        output_df = pd.DataFrame(output_data, columns=columns)

        # Write to file
        _ = "a" if i % 2 == 0 else "b"
        j = int(i / 2) if i % 2 == 0 else int((i - 1) / 2)
        save_path = pjoin(save_dir, "_".join(["template", str(j + 1), _]) + ".tsv")

        # Save DataFrame
        output_df.to_csv(save_path, sep="\t", index=False)


def add_adj_sentences(headers, row_orig, adjs, save_path, opp=False):

    save_path = save_path.format("_opp") if opp else save_path.format("_norm")

    output_data = []
    output_data.append(row_orig)  # first line is the original sentence w/o an adjective

    for adj in adjs:
        row = row_orig.copy()
        row['adjective'] = adj
        output_data.append(row)

    df = pd.DataFrame(data=output_data, index=None, columns=headers)
    df.to_csv(save_path, sep='\t', index=True)


def create_f_m_adj_sentences(templates_path, f_adj_path, m_adj_path):
    save_dir = pjoin(templates_path, "adjectives")
    if not isdir(save_dir):
        os.mkdir(save_dir)

    templates = sorted([pjoin(templates_path, f) for f in listdir(templates_path) if ".tsv" in f])
    female_adjectives = list(pd.read_csv(f_adj_path, sep="\t")["adjective"])
    male_adjectives = list(pd.read_csv(m_adj_path, sep="\t")["adjective"])

    for template in templates:
        base = os.path.basename(template).split(".")[0]
        f_save_path = pjoin(save_dir, base+"_female{}.tsv")
        m_save_path = pjoin(save_dir, base+"_male{}.tsv")

        df = pd.read_csv(template, sep="\t")
        headers = list(df.columns)
        row_orig = df.loc[4]
        row_orig_opp = df.loc[5]

        add_adj_sentences(headers, row_orig, female_adjectives, f_save_path)
        add_adj_sentences(headers, row_orig, male_adjectives, m_save_path)

        add_adj_sentences(headers, row_orig_opp, female_adjectives, f_save_path, opp=True)
        add_adj_sentences(headers, row_orig_opp, male_adjectives, m_save_path, opp=True)

def get_pronouns(sentences):
    pronouns = ["she", "her", "he", "his", "him", "they", "their", "them"]
    return [pronouns[i] for sent in sentences for i,p in enumerate(pronouns) if p in sent.lower().split(" ")]

def get_rows(_row, sentences, pronouns=True):
    occ, par, answer, gen_adj, verbs, sentence = _row
    if pronouns:
        pronouns = get_pronouns(sentences)
        return [[occ, par, answer, gen_adj, verbs, pronouns[i], m_sent, sentence] for i, m_sent in enumerate(sentences)]
    else:
        return [[occ, par, answer, gen_adj, verbs, '', m_sent, sentence] for m_sent in sentences]


def generate_sentence_from_template(occupation, participant, answer, adjective, verbs, sentence,
                                    be=False,
                                    someone=False,
                                    opp=False,
                                    mask_occupation=False,
                                    mask_participant=False,
                                    mask_adjective=False,
                                    mask_verb=False,
                                    mask_pronoun=False,
                                    multi_choice=False):
    if str(adjective) == "nan":
        adjective = ""
    toks = sentence.split(" ")

    if "PRONOUN was" in sentence:
        be = True

    # ----- pronouns

    NOM = "$NOM_PRONOUN"
    POSS = "$POSS_PRONOUN"
    ACC = "$ACC_PRONOUN"

    special_toks = set({NOM, POSS, ACC})

    # ----- occupation

    occ_index = toks.index("$OCCUPATION")

    # replace occupation placeholder or mask
    toks[occ_index] = occupation if not mask_occupation else "[MASK]"

    def remove_token(token, toks, reduce_indices=False, indices=None):
        if not reduce_indices and not indices:
            index = toks.index(token)
            return toks[:index] + toks[index + 1:]
        else:
            index = toks.index(token)
            new_indices = [i if i < index else i - 1 for i in indices]
            return toks[:index] + toks[index + 1:], new_indices

    # ----- 'someone'

    # replace a token with 'someone' or mask
    def insert_someone(toks, part_index, indices=None, mask=False):
        # first, remove the token that precedes $PARTICIPANT, i.e. "the"
        if part_index == 1:
            toks = toks[part_index:]
            if indices:
                indices = [i if i < part_index else i - 1 for i in indices]
        else:
            if toks[part_index - 1] == "$ADJECTIVE":
                toks = toks[:part_index - 2] + toks[part_index - 1:]
            else:
                toks = toks[:part_index - 1] + toks[part_index:]
            if indices:
                indices = [i if i < part_index else i - 1 for i in indices]

        # recompute participant index (it should be part_index - 1)
        part_index = toks.index("$PARTICIPANT")

        if part_index == 0:
            toks[part_index] = "Someone" if not mask else "[MASK]"
        else:
            toks[part_index] = "someone" if not mask else "[MASK]"

        if indices:
            return toks, indices
        else:
            return toks

    # ----- verb

    verb_indices = [toks.index(x) for x in toks if "$" in x and x not in special_toks]

    # ----- participant

    part_index = toks.index("$PARTICIPANT")

    # replace participant placeholder or mask
    if not someone:
        toks[part_index] = participant if not mask_participant else "[MASK]"
    # or replace participant with 'someone'
    else:
        if answer == 1 and not opp or answer == 0 and opp:
            toks, verb_indices = remove_token("$ADJECTIVE", toks, True, verb_indices)

        part_index = toks.index("$PARTICIPANT")
        if not mask_participant:
            toks, verb_indices = insert_someone(toks, part_index, verb_indices)
        else:
            toks, verb_indices = insert_someone(toks, part_index, verb_indices, mask=True)

    # ----- verb

    verb_indices = [toks.index(x) for x in toks if "$" in x and x not in special_toks]
    verbs = verbs.split(" ")

    # ----- someone & adjective

    if not someone:
        # replace adjective placeholder or mask
        # empty string if no adjective provided
        adj_index = toks.index("$ADJECTIVE")
        toks[adj_index] = adjective if not mask_adjective else "[MASK]"
    else:
        if answer == 0 or answer == 1 and opp:
            if answer == 0 and opp:
                pass
            else:
                adj_index = toks.index("$ADJECTIVE")
                toks[adj_index] = adjective if not mask_adjective else "[MASK]"

                # ----- remove empty string

    if "" in toks:
        empty_index = toks.index("")
        toks = toks[:empty_index] + toks[empty_index + 1:]

    # ----- verbs

    def mask_verb_func(toks, verb_indices, verbs, special_toks):
        if "" in verbs:
            empty_index = verbs.index("")
            verbs = verbs[:empty_index] + verbs[empty_index + 1:]

        if len(verbs) == 1:
            toks[verb_indices[0]] = "[MASK]"
            return toks

        # mask verbs and addons
        VBD = "$VERB_VBD"
        VBN = "$VERB_VBN"
        VBG = "$VERB_VBG"
        ADP = "$ADP_IN"
        PAR = "$PART_RP"

        placeholders = [toks[i] for i in verb_indices]

        num_VBD = [x for i, x in enumerate(placeholders) if x == VBD]

        # remove ADP_IN and PART_RP tokens
        if ADP in placeholders:
            ADP_tok = verbs[verb_indices.index(toks.index(ADP))]
            if ADP_tok in ["with", "in"]:  # remove preposition 'with' or 'in'
                toks = remove_token(ADP, toks)
            elif ADP_tok in ["to", "on"]:
                toks[toks.index(ADP)] = ADP_tok
        if PAR in placeholders:
            toks = remove_token(PAR, toks)

        # mask verb placeholder
        if len(num_VBD) > 1:
            # this skips sentences with more than 1 verb
            return
        elif VBG in placeholders:
            toks[toks.index(VBG)] = "[MASK]"
            toks = remove_token(VBD, toks)  # remove was
        elif len(num_VBD) == 1:
            if VBN in placeholders:
                toks[toks.index(VBN)] = "[MASK]"
                # toks = remove_token(VBD, toks)  # remove vs keep VBD
                toks[toks.index(VBD)] = verbs[verb_indices.index(toks.index(VBD))]  # keep VBD
            else:
                toks[toks.index(VBD)] = "[MASK]"
        return toks

    # replace verb placeholders
    verb_indices = [toks.index(x) for x in toks if "$" in x and x not in special_toks]

    if not mask_verb:
        # make sure there are as many verbs to be inserted as indices found
        try:
            assert (len(verbs) == len(verb_indices))
        except AssertionError as e:
            print("Assertion Error")
            print(verbs)
            print(verb_indices)
            print(toks)
            print(someone, mask_occupation, mask_participant,
                  mask_adjective, mask_verb, mask_pronoun)
            return
        for i, p in enumerate(verb_indices):
            toks[p] = verbs[i]
    else:
        toks = mask_verb_func(toks, verb_indices, verbs, special_toks)

    # ----- pronouns

    # pronoun form dictionary mapping
    female_map = {NOM: "she", POSS: "her", ACC: "her"}
    male_map = {NOM: "he", POSS: "his", ACC: "him"}
    neutral_map = {NOM: "they", POSS: "their", ACC: "them"}

    def map_toks(toks, special_toks, x_map, mask=False):
        if mask:
            return [x if not x in special_toks else "[MASK]" for x in toks]
        else:
            return [x if not x in special_toks else x_map[x] for x in toks]

    # Create multiple choice question style sentences
    if multi_choice:

        pronoun_idx = [toks.index(t) for i,t in enumerate(toks) if "PRONOUN" in t][0]
        # toks = toks[:pronoun_idx] + ["[SEP]"] + toks[pronoun_idx:]  # for sentence pairs
        toks = ["[CLS]"] + toks[:-1] + [toks[-1].split(".")[0]] + ["."] + ["[SEP]"]

    # replace pronouns with their respective forms or mask
    if not mask_pronoun:
        female_toks = map_toks(toks, special_toks, female_map)
        male_toks = map_toks(toks, special_toks, male_map)
        neutral_toks = map_toks(toks, special_toks, neutral_map)
    else:
        female_toks = map_toks(toks, special_toks, female_map, mask=True)
        male_toks = map_toks(toks, special_toks, male_map, mask=True)
        neutral_toks = map_toks(toks, special_toks, neutral_map, mask=True)

    female_sent = " ".join(female_toks)
    male_sent = " ".join(male_toks)
    neutral_sent = " ".join(neutral_toks)

    if not mask_pronoun:
        neutral_sent = neutral_sent.replace("they was", "they were")
        neutral_sent = neutral_sent.replace("They was", "They were")
    else:
        neutral_sent = neutral_sent.replace("[MASK] was", "[MASK] were")

    if not mask_pronoun:
        sentences = [female_sent, male_sent, neutral_sent]
    else:
        if be:
            assert (female_sent == male_sent)
            assert (female_sent != neutral_sent)
            sentences = [female_sent, neutral_sent]
        else:
            assert (female_sent == male_sent)
            assert (female_sent == neutral_sent)
            sentences = [neutral_sent]

    return sentences


def create_f_m_masked_adj_sentences(adj_templates_dir, project_dir):
    # Create directories
    mkdir(project_dir)
    for name in ["occupation", "participant", "adjectives", "verb", "pronouns"]:
        _dir = pjoin(project_dir, name)
        _dir_someone = pjoin(project_dir, name + "_someone")
        mkdir(_dir)
        mkdir(_dir_someone)

    # Save path
    save_dir_template = pjoin(project_dir, "{}")

    # List of all templates
    adj_templates = sorted([pjoin(adj_templates_dir, file) for file in listdir(adj_templates_dir)])

    # Create masked sentences
    for j, file in enumerate(adj_templates):
        df = pd.read_csv(file, sep="\t")
        cols = list(df.columns)
        new_cols = cols[1:-1] + ["pronoun", "masked_sentence"] + cols[-1:]

        gen = "f" if "female" in file else "m"
        opp = True if "_opp" in file else False

        masked_sentences = {"occupation": [], "occupation_someone": [],
                            "participant": [], "participant_someone": [],
                            "adjectives": [], "adjectives_someone": [],
                            "verb": [], "verb_someone": [],
                            "pronouns": [], "pronouns_someone": []}

        try:
            for i in range(df.shape[0]):
                row = df.iloc[i]

                occ = row['occupation(0)']
                par = row['other-participant(1)']
                answer = row['answer']
                adj = row['adjective']
                if str(adj) == 'nan':
                    adj = ''
                verbs = row['verb']
                sentence = row['sentence']

                gen_adj = "{}_{}".format(gen, adj) if adj != '' else ''

                _row = [occ, par, answer, gen_adj, verbs, sentence]

                # Mask occupations
                occ_mask_sents = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                 mask_occupation=True, opp=opp)
                new_rows = get_rows(_row, occ_mask_sents)
                masked_sentences["occupation"] = masked_sentences["occupation"] + new_rows
                # insert someone
                occ_mask_sents_someone = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                         someone=True, mask_occupation=True, opp=opp)
                new_rows = get_rows(_row, occ_mask_sents_someone)
                masked_sentences["occupation_someone"] = masked_sentences["occupation_someone"] + new_rows

                # Mask participants
                par_mask_sents = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                 mask_participant=True, opp=opp)
                new_rows = get_rows(_row, par_mask_sents)
                masked_sentences["participant"] = masked_sentences["participant"] + new_rows
                # insert someone
                par_mask_sents_someone = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                         someone=True, mask_participant=True, opp=opp)
                new_rows = get_rows(_row, par_mask_sents_someone)
                masked_sentences["participant_someone"] = masked_sentences["participant_someone"] + new_rows

                # Mask adjectives
                if i == 0 and "_male" in file:
                    adj_mask_sents = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                     mask_adjective=True, opp=opp)
                    new_rows = get_rows(_row, adj_mask_sents)
                    masked_sentences["adjectives"] = masked_sentences["adjectives"] + new_rows
                    # insert someone
                    adj_mask_sents_someone = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                             someone=True, mask_adjective=True, opp=opp)
                    new_rows = get_rows(_row, adj_mask_sents_someone)
                    masked_sentences["adjectives_someone"] = masked_sentences["adjectives_someone"] + new_rows


                # Mask verbs
                verb_mask_sents = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                  mask_verb=True, opp=opp)
                new_rows = get_rows(_row, verb_mask_sents)
                masked_sentences["verb"] = masked_sentences["verb"] + new_rows
                # insert someone
                verb_mask_sents_someone = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                          someone=True, mask_verb=True, opp=opp)
                new_rows = get_rows(_row, verb_mask_sents_someone)
                masked_sentences["verb_someone"] = masked_sentences["verb_someone"] + new_rows

                # Mask pronouns
                # if "PRONOUN was" in sentence:
                #     pronoun_mask_sent = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                #                                                         be=True, mask_pronoun=True, opp=opp)
                #     new_rows = get_rows(pronoun_mask_sent, pronouns=False)
                #     masked_sentences["pronouns"] = masked_sentences["pronouns"] + new_rows
                #     # someone
                #     pronoun_mask_sent_someone = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                #                                                                 be=True, someone=True, mask_pronoun=True,
                #                                                                 opp=opp)
                #     new_rows = get_rows(pronoun_mask_sent_someone, pronouns=False)
                #     masked_sentences["pronouns_someone"] = masked_sentences["pronouns_someone"] + new_rows
                # else:
                pronoun_mask_sent = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                    mask_pronoun=True, opp=opp)
                new_rows = get_rows(_row, pronoun_mask_sent, pronouns=False)
                masked_sentences["pronouns"] = masked_sentences["pronouns"] + new_rows
                # someone
                pronoun_mask_sent_someone = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                            someone=True, mask_pronoun=True, opp=opp)
                new_rows = get_rows(_row, pronoun_mask_sent_someone, pronouns=False)
                masked_sentences["pronouns_someone"] = masked_sentences["pronouns_someone"] + new_rows

                # Save file
                for k, v in masked_sentences.items():
                    save_name = os.path.basename(file)
                    save_path = pjoin(save_dir_template.format(k), save_name)

                    new_df = pd.DataFrame(v, index=None, columns=new_cols)
                    new_df.to_csv(save_path, sep="\t", index=True)

        except ValueError as e:
            print("ValueError: {} in file\n{}".format(e, file))


def create_multiple_choice_sentences(save_dir, templates_dir):
    templates = sorted([pjoin(templates_dir, f) for f in listdir(templates_dir)])

    for j,file in enumerate(tqdm(templates)):
        df = pd.read_csv(file, sep="\t")
        cols = list(df.columns)
        new_cols = cols[1:4] + ["someone"] + cols[4:] + ["female", "male", "neutral"]
        n_rows = df.shape[0]

        gen = "f" if "female" in os.path.basename(file) else "m"
        opp = True if "_opp" in os.path.basename(file) else False

        new_rows = []

        for i in range(n_rows):
            row = df.iloc[i]

            occ = row['occupation(0)']
            par = row['other-participant(1)']
            answer = row['answer']
            adj = row['adjective']
            if str(adj) == 'nan':
                adj = ''
            verbs = row['verb']
            sentence = row['sentence']

            gen_adj = "{}_{}".format(gen, adj) if adj != '' else ''

            for someone in [True, False]:
                f_sent, m_sent, n_sent = generate_sentence_from_template(occ, par, answer, adj, verbs, sentence,
                                                                         opp=opp, multi_choice=True, someone=someone)

                new_row = [occ, par, answer, someone, gen_adj, verbs, sentence, f_sent, m_sent, n_sent]
                new_rows.append(new_row)

            # break

        save_name = os.path.basename(file).replace("template_", "")
        save_path = pjoin(save_dir, save_name)

        new_df = pd.DataFrame(new_rows, index=None, columns=new_cols)
        new_df.to_csv(save_path, sep="\t", index=True)

        # break


def main():
    # Set directories
    data_dir = pjoin(pjoin(os.getcwd(), os.pardir, os.pardir), "data")
    wg_dir = pjoin(data_dir, "winogender-schemas", "data")  # Winogender dir
    # wb_dir = pjoin(data_dir, "WinoBias", "data")  # WinoBias dir

    wg_template_path = pjoin(wg_dir, "templates.tsv")
    save_dir = pjoin(os.path.dirname(wg_template_path), "templates")

    ### Create adjective and verb sentence templates
    create_df_with_placeholders(wg_template_path, save_dir)

    # Adjective lists
    f_adjectives_path = pjoin(wg_dir, "female_adjectives.tsv")
    m_adjectives_path = pjoin(wg_dir, "male_adjectives.tsv")
    templates_path = save_dir

    ### Create adjective sentences
    create_f_m_adj_sentences(templates_path, f_adjectives_path, m_adjectives_path)

    # Directories for masking
    project_dir = pjoin(data_dir, "winogender-adj-verb", "masked")
    adj_templates_dir = pjoin(save_dir, "adjectives")

    ### Create maksed adjective sentences
    create_f_m_masked_adj_sentences(adj_templates_dir, project_dir)

    # Directories for multiple choice
    multiple_choice_dir = pjoin(data_dir, "winogender-adj-verb", "multiple-choice")
    mkdir(multiple_choice_dir)

    ### Create multiple choice question sentences
    create_multiple_choice_sentences(multiple_choice_dir, adj_templates_dir)


if __name__ == '__main__':
    main()