import random
import copy
from pyscipopt import Model
from pyscipopt import Conshdlr
import pyscipopt

import ecole

import gzip
import pickle
import numpy as np
from pathlib import Path
import time
import os
import glob
import random

import ray
import psutil
NUM_CPUS = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=NUM_CPUS)
except RuntimeError:
    # already initialised ray in script calling dcn sim, no need to init again
    pass


param_list = ["presolving/stuffing/","presolving/dualsparsify/","presolving/sparsify/",
              "presolving/tworowbnd/","presolving/redvub/","presolving/implics/",
              "presolving/dualinfer/", "presolving/dualagg/", "presolving/domcol/", #neg
              "presolving/gateextraction/", "presolving/boundshift/", "presolving/convertinttobin/",
              "presolving/inttobinary/", "presolving/trivial/", 
              ]

def with_presolve(Problem):
    model = Model("test")
    model.hideOutput()
    model.setLogfile("log_optimized.txt")

    model.readProblem(Problem)

    model.writeProblem("before_presolve.lp")

    model.setParam("limits/time", 600)

    for i in range(len(param_list)):

        model.setParam(param_list[i] + "priority", 666666)
        model.setParam(param_list[i] + "maxrounds", 10)
        model.setParam(param_list[i] + "timing", 16)

    # 进行预处理
    model.presolve()

    # 保存预处理之后的模型
    model.writeProblem("after_presolve.lp", trans=True, genericnames=True)

    model.optimize()

    bestSol = model.getBestSol()
    objVal = model.getObjVal()
    print('with_presolve' + '-'*50)
    print(f'bestSolution: \n: {bestSol}')
    print(f'objVal: \n: {objVal}')
    
    return bestSol, objVal
def no_presolve(Problem):
    model = Model("test")
    model.hideOutput()
    model.setLogfile("log_optimized.txt")

    model.readProblem(Problem)

    model.setParam("limits/time", 600)

    # 保存预处理之后的模型
    #model.writeProblem("after_presolve.lp", trans=True, genericnames=True)

    model.optimize()

    bestSol = model.getBestSol()
    objVal = model.getObjVal()
    print('with_presolve' + '-'*50)
    print(f'bestSolution: \n: {bestSol}')
    print(f'objVal: \n: {objVal}')
    
    return bestSol, objVal


@ray.remote
# def run_sampler(co_class, branching, nrows, ncols, max_steps=None, instance=None):
def run_presolve(instance):

    model = Model("test")
    model.hideOutput()
    model.setLogfile("log_optimized.txt")

    model.readProblem(instance)

    model.setParam("limits/time", 600)

    for i in range(len(param_list)):

        model.setParam(param_list[i] + "priority", 666666)
        model.setParam(param_list[i] + "maxrounds", 10)
        model.setParam(param_list[i] + "timing", 16)

    # 进行预处理
    model.presolve()

    # 新文件路径
    new_file_path = get_presolve_path(instance)

    # 保存预处理之后的模型
    model.writeProblem(new_file_path, trans=True, genericnames=True)
    
    return None

def get_presolve_path(path):
    # 获取文件名和目录
    directory, filename = os.path.split(path)
    # 获取第一层文件夹
    first_folder = directory.split(os.sep)[-1]
    # 创建新路径
    new_first_folder = f'presolve_{first_folder}'
    new_directory = os.path.join(os.path.dirname(directory), new_first_folder)
    # 组合成新路径
    pre_solve_path = os.path.join(new_directory, filename)

    return pre_solve_path

if __name__ == "__main__":

    instances = iter(glob.glob(f'/data/ltf/code/retro_branching_offline/datasets/instances/setcover/train_500r_1000c_0.05d//*.lp'))

    result_ids = []
    for instance in instances:

        if  'presolve' in instance:
            continue
        
        result_ids.append(run_presolve.remote(instance = instance))

        if len(result_ids) >= int(NUM_CPUS* 2):
            # 等待任务完成并获取结果
            result = ray.get(result_ids)
            result_ids = []
            print(f'curr instance is : {instance}')

    print('done')
