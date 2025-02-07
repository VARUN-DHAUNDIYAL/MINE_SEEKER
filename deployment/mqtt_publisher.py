import paho.mqtt.client as mqtt
import json

MQTT_BROKER = "your-iot-endpoint.amazonaws.com"
MQTT_TOPIC = "landmine/detection"

client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

def publish_prediction(prediction):
    payload = json.dumps({"landmine_detected": prediction})
    client.publish(MQTT_TOPIC, payload)

# Example usage:
publish_prediction(True)
