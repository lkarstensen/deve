import numpy as np
import eve
import pygame
from eve.vesseltree import Vector
from time import perf_counter

vessel_tree = eve.vesseltree.AorticArch(rotate_z=20, rotate_x=-5, omit_y_axis=False)
device = eve.simulation3d.device.JWire(visu_edges_per_mm=0.5)
device2 = eve.simulation3d.device.JWire(
    name="cath",
    visu_edges_per_mm=0.5,
    tip_outer_diameter=1.2,
    straight_outer_diameter=1.2,
    tip_inner_diameter=1.0,
    straight_inner_diameter=1.0,
    color=[1.0, 0.0, 0.0],
)

# device2 = eve.simulation3d.device.Simmons3Bends()

simulation = eve.simulation3d.MultiDevice(
    vessel_tree=vessel_tree, devices=[device, device2]
)
start = eve.start.MaxDeviceLength(simulation, 380)
target = eve.target.CenterlineRandom(vessel_tree, simulation, threshold=10)
success = eve.success.TargetReached(target)
pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, simulation, target)

position = eve.observation.Tracking(simulation, n_points=5)
position = eve.observation.wrapper.RelativeToFirstRow(position)
position = eve.observation.wrapper.ToTrackingCS(position, dim_to_delete="y")
# position = eve.state.wrapper.Normalize(position)
target_state = eve.observation.Target(target)
target_state = eve.observation.wrapper.ToTrackingCS(target_state, dim_to_delete="y")
# target_state = eve.state.wrapper.Normalize(target_state)
rotation = eve.observation.Rotations(simulation)
state = eve.observation.ObsDict([position, target_state, rotation])

target_reward = eve.reward.TargetReached(target, factor=1.0)
# step_reward = eve.reward.Step(factor=-0.01)
path_delta = eve.reward.PathLengthDelta(pathfinder, 0.01)
reward = eve.reward.Combination([target_reward, path_delta])

max_steps = eve.terminal.MaxSteps(200)
target_reached = eve.terminal.TargetReached(target)
done = eve.terminal.Combination([max_steps, target_reached])

visualisation = eve.visualisation.SofaPygame(simulation)

randomizer = eve.vesseltreerandomizer.AorticArchRandom(
    vessel_tree, intervention=simulation, mode="eval"
)

env = eve.Env(
    vessel_tree=vessel_tree,
    intervention=simulation,
    start=start,
    target=target,
    success=success,
    observation=state,
    reward=reward,
    terminal=done,
    visualisation=visualisation,
    pathfinder=pathfinder,
    vessel_tree_randomizer=randomizer,
)

r_cum = 0.0

env.reset()
last_tracking = None
while True:
    start = perf_counter()
    trans = 0.0
    rot = 0.0
    camera_trans = Vector(0.0, 0.0, 0.0)
    camera_rot = Vector(0.0, 0.0, 0.0)
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
            camera_trans += Vector(0.0, 0.0, 200.0)
        if keys_pressed[pygame.K_s]:
            camera_trans -= Vector(0.0, 0.0, 200.0)
        if keys_pressed[pygame.K_a]:
            camera_trans -= Vector(200.0, 0.0, 0.0)
        if keys_pressed[pygame.K_d]:

            camera_trans = Vector(200.0, 0.0, 0.0)
        env.visualisation.translate(camera_trans)
    if keys_pressed[pygame.K_e]:
        env.visualisation.zoom(1000)
    if keys_pressed[pygame.K_q]:
        env.visualisation.zoom(-1000)

    # trans = 10
    if keys_pressed[pygame.K_v]:
        action = ((0, 0), (trans, rot))

    else:
        action = ((trans, rot), (0, 0))
    s, r, d, i, success = env.step(intervention_action=action)

    if keys_pressed[pygame.K_RETURN]:
        env.reset()
        n_steps = 0
    tracking = env.intervention.device_trackings
    tracking_2 = np.array(env.intervention.tracking)

    # print(tracking[0][0:5])
    # print(len(env.simulation.tracking_per_device[0]))
    # print(f"FPS: {1/(perf_counter()-start)}")
env.close()


# for _ in range(3):
#     print(env.reset())
#     for _ in range(10):
#         print(env.step(np.array([10, 0.5])))

# print("success")
