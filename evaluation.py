#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Joao Pedro <joao.gandarela@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
import glob
import params
import numpy as np
from pyswip import Prolog
from pyswip.prolog import PrologError
from sklearn.metrics import accuracy_score, precision_score, recall_score

def run(path: str, model: str = "gpt-4o"):
    results = {}
    for g in glob.glob(path):
        for c in glob.glob(f"{g}/*"):
            category = c.split("/")[-1]
            for f in glob.glob(f"{c}/*"):
                if category not in results: results[category] = {"acc":[],"time":[],"f1":[]}
                # print(f"category = {category} - {f.split('/')[-1]}")
                iteration = 1
                aacc = []
                f1s = []
                with open(f"{f}/{model.replace('/', '-')}_time_llm.txt", "r") as tt:
                        results[category]["time"].append(float(tt.read().strip()))
                while True:
                    y_true = []
                    y_pred = []
                    prolog = Prolog()
                    pt_theory_path = f"{f}/{model.replace('/', '-')}_pt{iteration}_theory.txt"
                    print(pt_theory_path)
                    if not os.path.isfile(pt_theory_path): break
                    prolog.consult(f"{f}/bk.pl")
                    prolog.consult(pt_theory_path)
                    with open(f"{f}/pos_test.pl", "r") as fi:
                        exs = fi.read().strip().split("\n")
                        for e in exs:
                            if len(e.strip()) == 0: continue
                            y_true.append(1)

                            try:
                                r = len(list(prolog.query(f"{e.strip()}")))
                            except PrologError:
                                r = 0

                            if r:
                                y_pred.append(1)
                            else:
                                y_pred.append(0)

                    with open(f"{f}/neg_test.pl", "r") as fi:
                        exs = fi.read().strip().split("\n")
                        for e in exs:
                            if len(e.strip()) == 0: continue
                            y_true.append(0)
                            try:
                                r = len(list(prolog.query(f"{e.strip()}")))
                            except PrologError:
                                r = 0

                            if not r:
                                y_pred.append(0)
                            else:
                                y_pred.append(1)

                    del prolog
                    acc = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    if precision+recall == 0: f1s.append(0)
                    else: f1s.append(2*(precision*recall)/(precision+recall))
                    aacc.append(acc)
                    iteration += 1
                if len(aacc):
                    results[category]["acc"].append(max(aacc))
                    results[category]["f1"].append(max(f1s))
    print(f"{model.upper()} - noise {path.split('_')[-2]}")
    for r in results:
        print(f"(Category {r}) - {np.mean(results[r]['f1']):.3f}\u00B1{np.std(results[r]['f1']):.2f} | {np.mean(results[r]['acc']):.3f}\u00B1{np.std(results[r]['acc']):.2f} | {np.mean(results[r]['time']):.3f}\u00B1{np.std(results[r]['time']):.2f}")

def main():
    for dec in [1, 2, 3]:
        run(params.D_PATH.format(dec=dec), "gpt-4o")


if __name__ == "__main__":
    main()
