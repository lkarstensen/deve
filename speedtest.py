from time import perf_counter
from eve.vesseltree.vmr import VMR
from eve.vesseltree.aorticarch import AorticArch
from eve.simulation3d.device.jwire import JWire
from eve.simulation3d.guidewire import Guidewire
from eve.simulation3d.multidevice import MultiDevice

from eve.simulation2d.device.jwire import JWire as JWire2D
from eve.simulation2d.singledevice import SingleDevice
from eve.visualisation.sofapygame import SofaPygame
from eve.visualisation.plt2D import PLT2D
import numpy as np


def speedtest(episodes: int, steps: int, sim: MultiDevice, rng, visu):
    performance = []
    sim.reset()
    sim.reset_devices()
    if visu is not None:
        visu.reset()

    for _ in range(episodes):
        action = [rng.uniform(10.0, 50.0), rng.uniform(0.0, 3.14)]
        start = perf_counter()
        for _ in range(steps):
            sim.step([action])
            if visu is not None:
                visu.step()
        sim.reset()
        sim.reset_devices()
        performance.append(perf_counter() - start)
        if visu is not None:
            visu.reset()
    return performance


vessel = AorticArch(scale_xyzd=[1.0, 1.0, 1.5, 0.8], seed=15)
vessel.reset()

device = JWire()
device_2d = JWire2D()

sim_gw = Guidewire(vessel, device)
sim_md = MultiDevice(vessel, [device])
sim_2d = SingleDevice(vessel, device_2d)
simulations = {"gw": sim_gw, "md": sim_md, "2d": sim_2d}


# visu_md = SofaPygame(sim_md)
# visu_gw = SofaPygame(sim_gw)
# visu_2d = PLT2D(vessel, sim_2d)
# visus = {"gw": visu_gw, "md": visu_md, "2d": visu_2d}


EPISODES = 100
STEPS = 100
performances = {}

for key in ["gw", "md"]:
    rng = np.random.default_rng(100)
    sim = simulations[key]
    visu = None  # visus[key]
    perf = speedtest(EPISODES, STEPS, sim, rng, visu)
    performances[key] = perf

    average = sum(perf) / len(perf)
    max_s = max(perf)
    min_s = min(perf)
    print(f"{key}: Average: {average}s, min: {min_s}s, max:{max_s}s")
    # visu.close()
print("md without extra tracking nodes")
for key, value in performances.items():
    average = sum(value) / len(value)
    max_s = max(value)
    min_s = min(value)

    print(f"{key}: Average: {average}s, min: {min_s}s, max:{max_s}s")
