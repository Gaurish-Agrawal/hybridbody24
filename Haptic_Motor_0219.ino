#include <Wire.h>
#include <WiFi.h>
#include "Adafruit_DRV2605.h"

// Wi-Fi Credentials
const char* ssid = "wurc2.4";
const char* password = "robotics@washu";

WiFiServer server(80);

#define TCA9548A_ADDR 0x70  // Default I2C address of the multiplexer

// Create a single DRV2605 instance (since we switch between channels)
Adafruit_DRV2605 drv;

// Direction mapping to motor index
const char* directions[] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};

void setup() {
  Serial.begin(115200);
  Wire.begin();  // Initialize I2C communication

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
  handleSerialInput();  // Check if a test command is entered in Serial Monitor
  handleWiFiRequests(); // Check for incoming HTTP requests
}

void handleWiFiRequests() {
  WiFiClient client = server.available(); // Listen for incoming clients
  if (client) {
    Serial.println("New client connected.");
    String request = client.readStringUntil('\r');
    Serial.println("Request: " + request);

    String direction = "";
    float intensity = 1.0; // Default intensity
    int motorIndex = -1;

    // Identify direction and corresponding motor index
    for (int i = 0; i < 8; i++) {
      if (request.indexOf(String("/") + directions[i]) != -1) {
        direction = directions[i];
        motorIndex = i;
        break;
      }
    }

    // Extract intensity if available
    int valueIndex = request.indexOf("value=");
    if (valueIndex != -1) {
      String intensityStr = request.substring(valueIndex + 6);
      intensity = intensityStr.toFloat(); // Convert to float
      intensity = constrain(intensity, 0.0, 1.0); // Ensure it's in range [0,1]
    }

    if (motorIndex != -1) {
      activateMotor(motorIndex, intensity);
    }

    client.flush();
    client.stop();
    Serial.println("Client disconnected.");
  }
}

void handleSerialInput() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // Read input from Serial Monitor
    input.trim();  // Remove any leading/trailing whitespace

    int motorIndex = -1;
    for (int i = 0; i < 8; i++) {
      if (input.equalsIgnoreCase(directions[i])) {
        motorIndex = i;
        break;
      }
    }

    if (motorIndex != -1) {
      Serial.print("Testing Motor for direction: ");
      Serial.println(directions[motorIndex]);
      activateMotor(motorIndex, 1.0);  // Use full intensity for testing
    } else {
      Serial.println("Invalid direction! Enter one of: N, NE, E, SE, S, SW, W, NW");
    }
  }
}

void selectI2CChannel(int channel) {
  if (channel < 0 || channel > 7) return; // Ignore invalid channels

  Wire.beginTransmission(TCA9548A_ADDR);
  Wire.write(1 << channel); // Select the correct I2C channel
  Wire.endTransmission();
}

void activateMotor(int motorIndex, float intensity) {
  Serial.print("Activating Motor ");
  Serial.print(motorIndex);
  Serial.print(" for direction: ");
  Serial.print(directions[motorIndex]);
  Serial.print(" with intensity: ");
  Serial.println(intensity);

  selectI2CChannel(motorIndex); // Enable the correct multiplexer channel

  if (!drv.begin()) {
    Serial.print("Motor ");
    Serial.print(motorIndex);
    Serial.println(" failed to initialize!");
    return;
  }

  drv.selectLibrary(1); // Use haptic library 1
  drv.setMode(DRV2605_MODE_INTTRIG); // Internal trigger mode

  uint8_t scaledEffect = map(intensity * 100, 0, 100, 1, 127); // Scale intensity

  drv.setWaveform(0, 47); // Set effect intensity
  drv.setWaveform(1, 0); // End waveform
  drv.go(); // Trigger the effect

  delay(500); // Brief delay for haptic effect
}
