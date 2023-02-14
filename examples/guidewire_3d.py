# pylint: disable=no-member

from time import perf_counter
import pygame
import numpy as np
import eve
import eve.simulation3d
import eve.visualisation

# vessel_tree = eve.vesseltree.AorticArchRandom(
#     mode="eval",
#     scale_diameter_array=[0.6],
#     rotate_z_deg_array=[-20],
#     rotate_x_deg_array=[-5],
#     omit_axis="y",
# )
vessel_tree = eve.vesseltree.VMR(
    "/Users/lennartkarstensen/stacie/eve/eve/vesseltree/data/vmr/0094_0001/",
    -10,
    -2,
    rotate_yzx_deg=[0, 110, 8],
)

device = eve.simulation3d.device.JWire()

simulation = eve.simulation3d.Guidewire(
    vessel_tree=vessel_tree,
    device=device,
    sofa_native_gui=False,
    dt_simulation=0.006,
    stop_device_at_tree_end=True,
)
start = eve.start.MaxDeviceLength(
    intervention=simulation,
    max_length=500,
)
target = eve.target.CenterlineRandom(
    vessel_tree=vessel_tree,
    intervention=simulation,
    threshold=10,
)
success = eve.success.TargetReached(target=target)
pathfinder = eve.pathfinder.BruteForceBFS(
    vessel_tree=vessel_tree,
    intervention=simulation,
    target=target,
)

position = eve.observation.Tracking(
    intervention=simulation,
    n_points=5,
)
position = eve.observation.wrapper.RelativeToFirstRow(position)
# position = eve.observation.wrapper.CoordinatesTo2D(
#     position,
#     dim_to_delete="y",
# )
# position = eve.state.wrapper.Normalize(position)
target_state = eve.observation.Target(target=target)
target_state = eve.observation.wrapper.CoordinatesTo2D(
    target_state,
    dim_to_delete="y",
)
# target_state = eve.state.wrapper.Normalize(target_state)
rotation = eve.observation.Rotations(intervention=simulation)

state = eve.observation.ObsDict(
    {"position": position, "target": target_state, "rotation": rotation}
)

target_reward = eve.reward.TargetReached(
    target=target,
    factor=1.0,
)
# step_reward = eve.reward.Step(factor=-0.01)
path_delta = eve.reward.PathLengthDelta(
    pathfinder=pathfinder,
    factor=0.01,
)
reward = eve.reward.Combination([target_reward, path_delta])


target_reached = eve.terminal.TargetReached(target=target)
max_steps = eve.truncation.MaxSteps(200)

visualisation = eve.visualisation.SofaPygame(intervention=simulation)


env = eve.Env(
    vessel_tree=vessel_tree,
    intervention=simulation,
    start=start,
    target=target,
    success=success,
    observation=state,
    reward=reward,
    terminal=target_reached,
    truncation=max_steps,
    visualisation=visualisation,
    pathfinder=pathfinder,
)

n_steps = 0
r_cum = 0.0

env.reset()

while True:
    start = perf_counter()
    trans = 0.0
    rot = 0.0
    camera_trans = np.array([0.0, 0.0, 0.0])
    camera_rot = np.array([0.0, 0.0, 0.0])
    zoom = 0
    pygame.event.get()
    keys_pressed = pygame.key.get_pressed()

    if keys_pressed[pygame.K_ESCAPE]:
        break
    if keys_pressed[pygame.K_UP]:
        trans += 25
    if keys_pressed[pygame.K_DOWN]:
        trans -= 25
    if keys_pressed[pygame.K_LEFT]:
        rot += 1 * 3.14
    if keys_pressed[pygame.K_RIGHT]:
        rot -= 1 * 3.14
    if keys_pressed[pygame.K_r]:
        lao_rao = 0
        cra_cau = 0
        if keys_pressed[pygame.K_d]:
            lao_rao += 10
        if keys_pressed[pygame.K_a]:
            lao_rao -= 10
        if keys_pressed[pygame.K_w]:
            cra_cau -= 10
        if keys_pressed[pygame.K_s]:
            cra_cau += 10
        env.visualisation.rotate(lao_rao, cra_cau)
    else:
        if keys_pressed[pygame.K_w]:
            camera_trans += np.array([0.0, 0.0, 200.0])
        if keys_pressed[pygame.K_s]:
            camera_trans -= np.array([0.0, 0.0, 200.0])
        if keys_pressed[pygame.K_a]:
            camera_trans -= np.array([200.0, 0.0, 0.0])
        if keys_pressed[pygame.K_d]:
            camera_trans = np.array([200.0, 0.0, 0.0])
        env.visualisation.translate(camera_trans)
    if keys_pressed[pygame.K_e]:
        env.visualisation.zoom(1000)
    if keys_pressed[pygame.K_q]:
        env.visualisation.zoom(-1000)
    action = (trans, rot)
    obs, reward, terminal, truncation, info = env.step(action=action)
    n_steps += 1

    print(obs)

    if keys_pressed[pygame.K_RETURN]:
        env.reset()
        n_steps = 0

    print(f"FPS: {1/(perf_counter()-start)}")
env.close()
