#include <Arduino.h>
#include "WiFi.h"
#include "camera_setup.h"
#include "web_server.h"

String lastPicturePath = ""; 

const char* ssid = "ESP32-CAM-Network";
const char* password = "123456789";

void setup() {
    Serial.begin(115200);

    // Initialize the camera
    startCamera();
    Serial.println("Camera initialized");

    // Connect to Wi-Fi 
    WiFi.softAP(ssid, password);
    Serial.println("WiFi AP started");

    // Initialize SD card 
    if (!SD_MMC.begin()) {
        Serial.println("SD Card Mount Failed");
        return;
    }

    // Check if the index.html file exists on the SD card
    if (!SD_MMC.exists("/index.html")) {
        Serial.println("index.html file not found on SD card!");
        return;
    }

    // Start the web server
    startCameraServer();
    Serial.println("Camera server started");
}

void loop() {
}
