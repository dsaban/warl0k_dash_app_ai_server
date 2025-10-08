
import json, threading, time
import paho.mqtt.client as mqtt

def make_client(client_id: str, host: str="localhost", port: int=1883, keepalive: int=60):
    c = mqtt.Client(client_id=client_id, clean_session=True, protocol=mqtt.MQTTv311)
    c.connect(host, port, keepalive)
    return c

class MqttApp(threading.Thread):
    def __init__(self, client_id, on_setup, on_loop=None, host="localhost", port=1883):
        super().__init__(daemon=True)
        self.client = make_client(client_id, host, port)
        self.on_setup = on_setup
        self.on_loop = on_loop

    def run(self):
        if self.on_setup:
            self.on_setup(self.client)
        self.client.loop_start()
        try:
            while True:
                if self.on_loop:
                    self.on_loop(self.client)
                time.sleep(0.1)
        finally:
            self.client.loop_stop()
            self.client.disconnect()
