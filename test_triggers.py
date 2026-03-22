import evdev

dev = evdev.InputDevice("/dev/input/event5")
print("Pull each trigger separately. Ctrl+C to stop.\n")

for event in dev.read_loop():
    if event.type == 3 and event.code != 0:
        print(f"code={event.code} value={event.value}")
