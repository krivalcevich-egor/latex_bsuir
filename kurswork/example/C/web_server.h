#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include "esp_http_server.h"
#include "FS.h"
#include "SD_MMC.h"

extern String lastPicturePath;  

// Function to read the HTML file from SD card
String loadHtmlFile(const char* path) {
    File file = SD_MMC.open(path);
    if (!file) {
        Serial.printf("Failed to open file: %s\n", path);
        return String();
    }
    
    String htmlContent;
    while (file.available()) {
        htmlContent += (char)file.read();  
    }
    file.close();
    return htmlContent;
}

// handler for css
esp_err_t css_handler(httpd_req_t *req) {
    File file = SD_MMC.open("/styles.css");  
    if (!file) {
        Serial.println("Failed to open styles.css");
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }
    
    httpd_resp_set_type(req, "text/css");  
    String cssContent;
    while (file.available()) {
        cssContent += (char)file.read();
    }
    file.close();
    
    httpd_resp_send(req, cssContent.c_str(), cssContent.length());
    return ESP_OK;
}

// Handler for the index page
esp_err_t index_handler(httpd_req_t *req) {
    String htmlPage = loadHtmlFile("/index.html");
    
    if (htmlPage.isEmpty()) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, htmlPage.c_str(), htmlPage.length());
    return ESP_OK;
}

// Handler for video streaming
esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len;
    uint8_t * _jpg_buf;
    char part_buf[64];

    static const char* _STREAM_CONTENT_TYPE = 
                "multipart/x-mixed-replace;boundary=frame";
    static const char* _STREAM_BOUNDARY = "--frame";
    static const char* _STREAM_PART = 
            "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            res = ESP_FAIL;
        } else {
            _jpg_buf_len = fb->len;
            _jpg_buf = fb->buf;

            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY,
                                     strlen(_STREAM_BOUNDARY));
            if (res == ESP_OK) {
                size_t hlen = snprintf(part_buf, 64, _STREAM_PART, 
                                                    _jpg_buf_len);
                res = httpd_resp_send_chunk(req, part_buf, hlen);
            }
            if (res == ESP_OK) {
                res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, 
                                                            _jpg_buf_len);
            }
            if (res != ESP_OK) {
                break;
            }
            esp_camera_fb_return(fb);  
        }
    }

    return res;
}

// Handler for capturing a photo
esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t * fb = esp_camera_fb_get();  // Capture frame
    if (!fb) {
        Serial.println("Camera capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // Save frame to SD card
    String path = "/photo" + String(millis()) + ".jpg";
    File file = SD_MMC.open(path, FILE_WRITE);
    if (!file) {
        Serial.println("Failed to open file in write mode");
        esp_camera_fb_return(fb);  
        return ESP_FAIL;
    }

    file.write(fb->buf, fb->len);  // Write JPEG buffer to SD card
    file.close();
    Serial.printf("Photo saved to: %s\n", path.c_str());

    lastPicturePath = path;  

    esp_camera_fb_return(fb);  

    // Reinitialize the camera after capture 
    esp_camera_deinit();
    startCamera();

    // Reload page
    httpd_resp_set_type(req, "text/html");
    String htmlPage = loadHtmlFile("/reload.html");
    httpd_resp_send(req, htmlPage.c_str(), htmlPage.length());
    
    return ESP_OK;
}

// Start the HTTP server
void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t capture_uri = {
        .uri       = "/capture",
        .method    = HTTP_GET,
        .handler   = capture_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t css_uri = {
        .uri       = "/styles.css",
        .method    = HTTP_GET,
        .handler   = css_handler,
        .user_ctx  = NULL
    };

    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_register_uri_handler(server, &index_uri);
        httpd_register_uri_handler(server, &stream_uri);
        httpd_register_uri_handler(server, &capture_uri);
        httpd_register_uri_handler(server, &css_uri);
    }

}

#endif 
