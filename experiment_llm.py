#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Joao Pedro <joao.gandarela@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
# -*- coding: utf-8 -*-

# https://launchpad.net/~swi-prolog/+archive/ubuntu/stable
# !add-apt-repository -y ppa:swi-prolog/stable &> /dev/null
# !apt update &> /dev/null
# !apt install swi-prolog
# !pip install git+https://github.com/yuce/pyswip@master#egg=pyswip &> /dev/null
#
# !pip install --upgrade openai

import os
import time
import re
import glob
import json
import params
from pyswip import Prolog
from pyswip.prolog import PrologError
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from openai import AzureOpenAI
from llm_local import LocalLLM

PROMPT_TEMPLATES = json.load(open("prompt_templates.json"))
llm_model = None

def remove_pos_neg_theory(rule):
    # Remove `pos` and `neg` predicates
    rule = re.sub(r'\bpos\((.*?)\)\s*,?\s*', r'\1, ', rule)
    rule = re.sub(r'\bpositive\((.*?)\)\s*,?\s*', r'\1, ', rule)
    rule = re.sub(r'\bneg\((.*?)\)\s*,?\s*', r'\1, ', rule)
    rule = re.sub(r'\bnegative\((.*?)\)\s*,?\s*', r'\1, ', rule)
    rule = re.sub(r'\btheory\((.*?)\)\s*,?\s*', r'\1', rule)
    # Remove trailing comma if it exists
    rule = re.sub(r',\s*\.', r'.', rule)
    return rule

def balance_parentheses(rule):
    # Extract the head and body of the rule
    head, body = re.match(r'(.*?) *:- *(.*)', rule, re.DOTALL).groups()
    print(f"head = {head}")
    print(f"body = {body}")
    # Count opening and closing parentheses in the body
    open_parens = body.count('(')
    close_parens = body.count(')')
    
    # Add missing closing parentheses if needed
    if open_parens > close_parens:
        body += ')' * (open_parens - close_parens)
    elif open_parens < close_parens:
        body = '(' + body * (close_parens - open_parens)
    
    # Count opening and closing parentheses in the body
    open_parens = head.count('(')
    close_parens = head.count(')')
    # Add missing closing parentheses if needed
    if open_parens > close_parens:
        head += ')' * (open_parens - close_parens)
    if head[0] != "p":
        head = "p"+head
    return f"{head}:-{body}"

def evaluate(examples, f, file_name):
    prolog = Prolog()
    prolog.consult(f"{f}/bk.pl")
    prolog.consult(f"{f}/{file_name}")
    acc = []
    gacc = []
    wrong = []
    for e in tqdm(examples):
        e = e.strip()
        if not len(e): continue
        if "pos" in e:
            a = e
            e = re.findall(r"pos\((.*?\))\)",e)[0]
            # print(e)
            gacc.append(1)
            try:
                r = len(list(prolog.query(f"{e.strip()}")))
            except PrologError:
                r = 0
            if r:
                acc.append(1)
            else:
                acc.append(0)
                wrong.append(a.replace(".",""))
        else:
            e = re.findall(r"neg\((.*?\))\)",e)[0]
            gacc.append(0)
            try:
                r = len(list(prolog.query(f"{e.strip()}")))
            except PrologError:
                r = 0
            if not r:
                acc.append(0)
            else:
                acc.append(1)
                wrong.append(a.replace(".", ""))

    return gacc, acc, wrong


def run(path: str, model: str = "gpt-35-turbo", local: bool = False):
    global llm_model

    if (local):
        client = LocalLLM(model) if not llm_model else llm_model
        llm_model = client
    else:
        openai_creds = json.load(open("openai_creds.json"))
        client = AzureOpenAI(**openai_creds)

    for g in glob.glob(path):
        for c in glob.glob(f"{g}/*"):
            category = c.split("/")[-1]
            # if category not in ["DISJUNCTIVE_ROOTED_DAG_RECURSIVE","MIXED","ROOTED_DAG","ROOTED_DAG_RECURSIVE"]: continue
            for f in tqdm(glob.glob(f"{c}/*"), desc=f"Processing {c}"):
                print(f"{category} - {f.split('/')[-1]}")
                start = time.time()
                iteration = 1
                bk = open(f"{f}/bk.pl")
                exs = open(f"{f}/exs.pl")
                user_input = PROMPT_TEMPLATES["input"].format(bk=bk.read().strip(), exs=exs.read().strip())
                bk.close()
                exs.close()
                if (model.startswith("gpt") or model.startswith("meta-llama")):
                    messages = [
                        {"role": "system",
                         "content": "Induce a theory based on background knowledge, positive and negative examples. Write it in prolog. Do not give an explanation. Answer only the theory."},
                        {"role": "user", "content": user_input}
                    ]
                else:
                    messages = [
                        {"role": "user",
                         "content": f"Induce a theory based on background knowledge, positive and negative examples. Write it in prolog. Do not give an explanation. Answer only the theory.\n\n{user_input}"}
                    ]
                result = 0
                while params.MAX_ITER >= iteration and params.M_THRESH > result:
                    # print(iteration)
                    # print(f)

                    pt_path = f"{f}/{model.replace('/', '-')}_pt{iteration}.txt"
                    if os.path.isfile(pt_path):
                        with open(pt_path, "r") as temp:
                            response = temp.read().strip()
                    else:
                        if (local):
                            response = client.prompt(messages, 500)
                        else:
                            response = client.chat.completions.create(
                                model=model,
                                messages=messages
                            ).choices[0].message.content
                        with open(pt_path, "w") as pt:
                            pt.write(response)

                    _theory = open(pt_path)
                    models_response = _theory.read().strip()
                    messages.append({"role": "assistant", "content": models_response})

                    rules = []
                    for r in re.findall(r"(?s)(\w+\s*\(.*?\)\s*:-\s*.*?\.)", models_response):
                        r = re.sub(r' *\n*\t*', '', r)
                        r = r.strip()
                        if not len(r): continue
                        r = balance_parentheses(remove_pos_neg_theory(r))
                        if len(re.findall(r"(?s)(\w+\s*\(.*?\)\s*:-\s*.*?\.)", r)):
                            head = re.findall(r"(.*)\(", r.split(":-")[0].strip())
                            if not len(head): continue
                            head = head[0]
                            tail = re.findall(r"[a-z]+\d*\([^)]+\)", r.split(":-")[1].strip())
                            recc = False
                            for ta in tail:
                                if head.strip() == re.findall(r"(.*)\(", ta.strip())[0]:
                                    recc = True
                                    break
                            if recc: continue
                            rules.append(r)

                    pt_thoery_path = f"{f}/{model.replace('/', '-')}_pt{iteration}_theory.txt"
                    with open(pt_thoery_path, "w") as pt:
                        pt.write("\n".join(rules))

                    with open(f"{f}/exs.pl") as eexs:
                        examples = eexs.read().strip().split("\n")

                    gacc, acc, wrong = evaluate(examples, f, f"{model.replace('/', '-')}_pt{iteration}_theory.txt")
                    prf = precision_recall_fscore_support(gacc, acc, average='binary')
                    result = accuracy_score(gacc, acc)

                    next = PROMPT_TEMPLATES["next"].format(
                        result=result,
                        precision=prf[0],
                        recall=prf[1],
                        f1=prf[2],
                        wrong=', '.join(wrong)
                    )

                    with open(f"{f}/{model.replace('/', '-')}_next{iteration}.txt", "w") as nx:
                        nx.write(next)

                    messages.append({"role": "user", "content": next})

                    n_results = 0
                    if (os.path.exists(f"{f}/{model.replace('/', '-')}_results.jsonl")):
                        with(open(f"{f}/{model.replace('/', '-')}_results.jsonl")) as results_file:
                            n_results = len(results_file.read().split("\n"))

                    if (n_results <= params.MAX_ITER):
                        with (open(f"{f}/{model.replace('/', '-')}_results.jsonl", "a")) as results_file:
                            results_file.write(json.dumps({
                                "acc": result,
                                "precision": prf[0],
                                "recall": prf[1],
                                "f1": prf[2],
                                "time": time.time() - start
                            }) + "\n")

                    iteration += 1

                llm_time_path = f"{f}/{model.replace('/', '-')}_time_llm.txt"
                if not os.path.isfile(llm_time_path):
                    with open(llm_time_path, "w") as timellm:
                        timellm.write(f"{time.time() - start}")


def main():
    for dec in [1, 2, 3]:
        run(params.D_PATH.format(dec=dec), "meta-llama/Meta-Llama-3-8B-Instruct", local=True)


if __name__ == "__main__":
    main()
