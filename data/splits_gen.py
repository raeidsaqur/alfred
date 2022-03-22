#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: splits_gen.py
# Created by: raeidsaqur on 2021-07-12
# Email: raeidsaqur@cs.toronto.edu
# 
# This file is part of embodied-ai
# Distributed under terms of the MIT License

import os
import sys
import json
import pprint
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import *

SPLIT_TYPES = ['tests_seen',
              'tests_unseen',
              'train',
              'valid_seen',
              'valid_unseen']

TASK_TYPES = ['pick_and_place_simple',
              'pick_clean_then_place_in_recep',
              'pick_heat_then_place_in_recep',
              'pick_cool_then_place_in_recep',
              'pick_two_obj_and_place',
              'look_at_obj_in_light',
              'pick_and_place_with_movable_recep']

print_count = lambda x: pprint.pprint({k: len(v) for k, v in x.items()})
get_task_id = lambda t: t.split("/")[-1]

def get_unique_task_ids_by_task_type(task_type:str, splits:Dict) -> Set:
      splits_task = get_splits_by_task_type(task_type, save_json=False, debug=False)
      taskids = set()
      for k,v in splits_task.items:
        for d in v:
            taskids.add(get_task_id(d['task']))

      return taskids

def get_splits_by_task_type(task_type:str,
                            proj_dir=None,
                            proj_name='alf', default_split="splits/oct21.json",
                            save_json=False,
                            debug=False) -> Dict:
    print(f"Getting splits by task: {task_type}")
    if not proj_dir:
        proj_dir = f"{os.path.expanduser('~')}/Research/projects/embodied-ai/{proj_name}"
        if not os.path.exists(proj_dir):
            raise FileNotFoundError("Project dir does not exist")

    splits_path = os.path.join(f"{proj_dir}/data", default_split)
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        if debug: print_count(splits)
    splits_task = {}
    for k,v in splits.items():
        if 'tests' in k:
            continue
        tvs = list(filter(lambda d: task_type in d['task'], v))
        if debug: print(f"No. of {task_type} samples in {k} = {len(tvs)}")
        splits_task[k] = tvs

    for stype in SPLIT_TYPES:
        # Set empty list for 'tests_[seen|unseen]'
        if stype not in splits_task.keys():
            splits_task[stype] = []

    if debug: print_count(splits_task)

    if save_json:
        fn_prefix = default_split.split("/")[-1].split('.')[0]
        fn = "{}-{}.json".format(fn_prefix, task_type)
        save_path = os.path.join(proj_dir, f"data/splits/{fn}")
        print(f"\tSaving {task_type} split to: {save_path}")
        with open(save_path, 'w') as fp:
            json.dump(splits_task, fp, sort_keys=True, indent=4)

    return splits_task

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--proj_dir', help="project folder")
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='splits/oct21.json')
    parser.add_argument('--proj_name', type=str, help="project name", default='alf')
    parser.add_argument('--task_type', help="1 of the 7 task types", default='pick_and_place_simple')
    parser.add_argument('--save_json', help="save gen. file", action='store_true')
    args = parser.parse_args()

    # splits_task = get_splits_by_task_type(args.task_type,
    #                                       proj_name=args.proj_name,
    #                                       debug=False,
    #                                       save_json=True)

    for task in TASK_TYPES:
        if 'pick_and_place_simple' in task:
            continue
        get_splits_by_task_type(task, save_json=True )


