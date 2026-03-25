
from hsm.demo_hsm import DemoHSM
from hub.controller import Hub

hsm = DemoHSM(b"root")
hub = Hub(hsm)

g = hub.create([], b"genesis", {})
a = hub.create([g], b"A", {})
b = hub.create([g], b"B", {})
m = hub.create([a,b], b"merge", {})

print("Validate:", hub.validate(m))
