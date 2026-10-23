# LUMEN – LDR Light Monitor

An ESP32/ESP8266 reads light with an LDR, uploads it to ThingSpeak, and a Streamlit dashboard displays the readings and predicts future values with linear regression.

```
LDR → ESP32/ESP8266 → ThingSpeak → Streamlit dashboard
```

## Files

| File | Purpose |
|---|---|
| `ldr_thingspeak.ino` | ESP sketch: reads LDR, uploads every 20 s |
| `app.py` | Streamlit dashboard with forecast |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark theme |

## Wiring

```
3.3V ──[ LDR ]──┬──[ 10kΩ ]── GND
                └── ESP32: GPIO34  |  ESP8266: A0
```

Do not power the divider from 5V.

## Setup

1. **ThingSpeak:** create a channel and enable **Field 1**. Note the Channel ID, Write API Key and Read API Key.
2. **ESP:** open `ldr_thingspeak.ino` in the Arduino IDE, set `WIFI_SSID`, `WIFI_PASSWORD` and `WRITE_API_KEY`, select your board, and upload. Serial Monitor (115200 baud) should show `Uploaded, entry #...`.
3. **Dashboard:**
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. In the sidebar, enter the Channel ID and Read API Key, pick field 1, and set the ADC range (4095 for ESP32, 1023 for ESP8266).

## Notes

- ThingSpeak free tier accepts one update per 15 s minimum.
- Linear regression suits short-term trends only; daylight is curved over a full day.
- LDR readings are relative, not calibrated lux.
