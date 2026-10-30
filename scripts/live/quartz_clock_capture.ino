// Companion sketch for scripts/live/quartz_clock_capture.py
// Streams the MCU's free-running tick counter once per second.
// The counter VALUES carry the oscillator phase — UART timing
// jitter does not enter the measurement.  Flash on APM 2.5
// (ATmega2560) or any Arduino; connect via USB serial.
//
// micros() wraps at 2^32 us (~71.6 min); the capture script
// unwraps on the host side — do NOT change to millis() unless
// you also remove the unwrap (it would double-count wraps).
// For runs > 49 days, millis() alone is the safer choice.

void setup() {
  Serial.begin(115200);
}

void loop() {
  Serial.println(micros());
  delay(1000);
}
