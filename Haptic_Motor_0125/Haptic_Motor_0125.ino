#include <WiFi.h>
#include "Adafruit_DRV2605.h"

Adafruit_DRV2605 drv;

// Wi-Fi credentials
const char* ssid = "wurc2.4";
const char* password = "robotics@washu";

WiFiServer server(80);

void setup() {
  Serial.begin(115200);

  if (!drv.begin()) {
    Serial.println("Could not find DRV2605");
    while (1) delay(10);
  }

  drv.selectLibrary(1); // Use haptic library 1
  drv.setMode(DRV2605_MODE_INTTRIG); // Internal trigger mode

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.println("Connecting to Wi-Fi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi. IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
}

void loop() {
  WiFiClient client = server.available(); // Listen for incoming clients
  if (client) {
    Serial.println("New client connected.");
    String request = client.readStringUntil('\r');
    Serial.println("Request: " + request);

    if (request.indexOf("/LEFT") != -1) {
      activateMotor("LEFT", 47); // Effect ID for LEFT
    } else if (request.indexOf("/RIGHT") != -1) {
      activateMotor("RIGHT", 10); // Effect ID for RIGHT
    } else if (request.indexOf("/UP") != -1) {
      activateMotor("UP", 17); // Effect ID for UP
    } else if (request.indexOf("/DOWN") != -1) {
      activateMotor("DOWN", 24); // Effect ID for DOWN
    }

    client.flush();
    client.stop();
    Serial.println("Client disconnected.");
  }
}

void activateMotor(String direction, uint8_t effect) {
  Serial.println("Activating motor for direction: " + direction);

  // Set different haptic patterns based on the effect ID
  drv.setWaveform(0, effect); // Play the effect
  drv.setWaveform(1, 0);      // End waveform
  drv.go();                   // Trigger the effect

  // Wait before allowing another trigger
  delay(500);
}
