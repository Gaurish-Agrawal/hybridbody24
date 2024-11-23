#include <WiFi.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"

Adafruit_DRV2605 drv;

// Wi-Fi credentials
const char* ssid = "wurc2.4";
const char* password = "robotics@washu";

// Create Wi-Fi server
WiFiServer server(80);

void setup() {
  Serial.begin(115200);
  Serial.println("Connecting to Wi-Fi...");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi.");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();

  // Initialize DRV2605
  if (!drv.begin()) {
    Serial.println("Could not find DRV2605");
    while (1) delay(10);
  }
  drv.selectLibrary(1); // Use library 1 for standard effects
  drv.setMode(DRV2605_MODE_INTTRIG); // Internal trigger mode
}

// Function to handle haptic feedback for "LEFT" and "RIGHT"
void activateEffect(String command) {
  if (command == "LEFT") {
    Serial.println("Haptic: Single Beat (LEFT)");
    drv.setWaveform(0, 1); // Strong click
    drv.setWaveform(1, 0); // End waveform
    drv.go();
    delay(500); // Pause to complete the effect
  } else if (command == "RIGHT") {
    Serial.println("Haptic: Max Intensity Double Beat (RIGHT)");
    drv.setWaveform(0, 14); // Strong buzz effect (max intensity)
    drv.setWaveform(1, 14); // Strong buzz again
    drv.setWaveform(2, 0);  // End waveform
    drv.go();
    delay(1000); // Increased delay for the "RIGHT" effect
  }
}

void loop() {
  // Check for client connection
  WiFiClient client = server.available();
  if (client) {
    Serial.println("New client connected.");
    String request = client.readStringUntil('\r'); // Read HTTP request
    Serial.println("Request: " + request);

    // Parse direction commands from request
    if (request.indexOf("/LEFT") != -1) {
      activateEffect("LEFT");
    } else if (request.indexOf("/RIGHT") != -1) {
      activateEffect("RIGHT");
    }

    client.flush();   // Clear the request
    client.stop();    // Close the connection
    Serial.println("Client disconnected.");
  }
}
