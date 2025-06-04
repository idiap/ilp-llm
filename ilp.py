#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Joao Pedro <joao.gandarela@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import os
import time
import glob
from tqdm import tqdm
from multiprocessing import Pool
from popper.loop import learn_solution
from popper.util import Settings, format_prog, order_prog
from sklearn.metrics import accuracy_score, precision_score, recall_score
import params
# from popper.util import print_prog_score

os.environ['PATH'] += f":{os.getcwd()}/WMaxCDCL/code/simp/"
os.environ['PATH'] += f":{os.getcwd()}/NuWLS-c-2023/bin/"

def run(f):
    fw_test = f"{f}/result_test_train_popper_{params.SOLVER}_{params.A_SOLVER}_{params.TIMEOUT}seg.txt"
    fw_train = f"{f}/result_train_popper_{params.SOLVER}_{params.A_SOLVER}_{params.TIMEOUT}seg.txt"
    fw_rule = f"{f}/rule_popper_{params.SOLVER}_{params.A_SOLVER}_{params.TIMEOUT}seg.pl"
    if os.path.isfile(fw_test) and os.path.isfile(fw_train):
        return
    try:
        print(f)
        start = time.time()
        settings = Settings(kbpath=f'{f}',anytime_solver=params.A_SOLVER,solver=params.SOLVER,timeout=params.TIMEOUT)
        prog, score, stats = learn_solution(settings)
    except Exception as e:
        print(e)
        return
    if prog != None:
        end = time.time()
        # print_prog_score(prog, score, False)
        tp, fn, tn, fp, size = score
        results = open(fw_train, "w")
        results.write(f"tp: {tp}\n")
        results.write(f"fn: {fn}\n")
        results.write(f"tn: {tn}\n")
        results.write(f"fp: {fp}\n")
        results.write(f"precision: {tp / (tp+fp):0.2f}\n")
        results.write(f"recall: {tp / (tp+fn):0.2f}\n")
        results.write(f"acc: {(tp+tn)/(tp+fp+tn+fn):0.2f}\n\n\n")
        print(f"rule:\n{format_prog(order_prog(prog))}")
        results.write(f"rule:\n{format_prog(order_prog(prog))}")
        tule = open(fw_rule, "w")
        tule.write(format_prog(order_prog(prog)))
        tule.close()
        from pyswip import Prolog
        prolog = Prolog()
        prolog.consult(fw_rule)
        prolog.consult(f"{f}/bk.pl")
        y_true = []
        y_pred = []
        with open(f"{f}/pos_test.pl", "r") as fi:
            exs = fi.read().strip().split("\n")
            for e in exs:
                e = e.strip()
                if not len(e): continue
                y_true.append(1)
                if len(list(prolog.query(f"{e}"))):
                    y_pred.append(1)
                else:
                    y_pred.append(0)
        with open(f"{f}/neg_test.pl", "r") as fi:
            exs = fi.read().strip().split("\n")
            for e in exs:
                e = e.strip()
                if len(e) == 0: continue
                y_true.append(0)
                if not len(list(prolog.query(f"{e}"))):
                    y_pred.append(0)
                else:
                    y_pred.append(1)
        del prolog
        results_test = open(fw_test, "w")
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        results_test.write(f"precision: {precision:0.2f}\n")
        results_test.write(f"recall: {recall:0.2f}\n")
        results_test.write(f"acc: {acc:0.2f}\n\n\n")
        if precision+recall == 0: results_test.write(f"f1: 0\n\n\n")
        else: 
            f1 = 2*(precision*recall)/(precision+recall)
            results_test.write(f"f1: {f1:0.2f}\n\n\n")
        results_test.write(f"time: {end - start}\n\n\n")
    else:
        results_test = open(fw_test, "w")
        results = open(fw_train, "w")
        results.write(f"sNO RULE WERE FOUND!!")
        results_test.write(f"NO RULE WERE FOUND!!")
        print("NO RULE WERE FOUND!!!")
    results.close()
    results_test.close()


pool = Pool(processes=1)

for dec in [1,2,3]:
    for g in tqdm(glob.glob(params.D_PATH.format(dec=dec))):
        for c in tqdm(glob.glob(f"{g}/*")):
            pool.map(run, glob.glob(f"{c}/*"))
            
pool.close()
pool.join()