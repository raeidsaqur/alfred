"""
Minimal example of spot where agent can walk into spot, but cannot look down &
look back up. Specifically: look_at_obj_in_light-Box-None-FloorLamp-212/trial_T20190908_193427_340509

Run from the root directory.
i.e.:

 export PYTHONPATH="."
 python3 ian/min/minimal.py



NOTE: Had to modify env/tasks.py to fully specify "from gen.graph"
      rather than "from graph"

      Had to modify gen/graph/graph_obj.py to fully specify "import gen.constants as constants"
      rather than "import constants"

      Had to modify X_DISPLAY = '1' rather than '0' in gen.constants
"""

import json
from env.thor_env import ThorEnv
from argparse import Namespace
import time

# Agent can freely spin in place at either of the end angles, set to True to demonstrate
SPIN = False
# Set to True to print shortened version of last event metadata (evidently,
# LookUp is translated into TeleportFull at some point)
PRINT_EVENT = False

def setup_scene(env, traj_data, r_idx, reward_type='dense'):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))

    # print goal instr
    print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    # setup task for reward
    args = Namespace(reward_config='models/config/rewards.json')
    env.set_task(traj_data, args, reward_type=reward_type)

if __name__ == "__main__":
    with open('ian/min/ann_0.json', 'r') as infile:
        traj = json.load(infile)

    env = ThorEnv()

    # setup scene
    reward_type = 'dense'
    setup_scene(env, traj, 0, reward_type=reward_type)

    # Run execute 21 expert actions
    expert_actions = [a['discrete_action'] for a in traj['plan']['low_actions']]
    for timestep in range(21):
        action_t = expert_actions[timestep]
        compressed_mask = action_t['args']['mask'] if 'mask' in action_t['args'] else None
        mask = env.decompress_mask(compressed_mask) if compressed_mask is not None else None
        action_t = action_t['action']

        print(timestep)
        print(action_t)
        t_success, _, _, err, _ = env.va_interact(action_t, interact_mask=mask, smooth_nav=False, debug=False)
        print(t_success)
        print(err)
        time.sleep(0.25)

    print("Starting new action sequence")
    time.sleep(2)

    if SPIN:
        # Spin in place; this suceeds
        for i in range(4):
            print("RIGHT TURN")
            t_success, _, _, err, _ = env.va_interact("RotateRight_90", interact_mask=None, smooth_nav=False, debug=False)
            print(t_success)
            print(err, flush=True)
            time.sleep(2)

    # Look down
    print("look down")
    t_success, _, _, err, _ = env.va_interact("LookDown_15", interact_mask=None, smooth_nav=False, debug=False)
    print(t_success)
    print(err, flush=True)
    time.sleep(2)

    if SPIN:
        # Spin in place; this suceeds
        for i in range(4):
            print("RIGHT TURN")
            t_success, _, _, err, _ = env.va_interact("RotateRight_90", interact_mask=None, smooth_nav=False, debug=False)
            print(t_success)
            print(err, flush=True)
            time.sleep(2)

    # Try to look up back to original position
    print("look up")
    t_success, new_event, _, err, _ = env.va_interact("LookUp_15", interact_mask=None, smooth_nav=False, debug=False)
    print(t_success)
    print(err, flush=True)

    if PRINT_EVENT:
        print("\n\n**************\n\n")
        if not new_event.metadata['lastActionSuccess']:
            import copy
            tmp = copy.deepcopy(new_event.metadata)
            del tmp['colors']
            del tmp['objects']
            del tmp["colorBounds"]
            import json
            print(json.dumps(tmp, indent=4))

    env.stop()





