import board
import pwmio
import math
import time

LEFT_ANTENNA_PIN = board.D13
RIGHT_ANTENNA_PIN = board.D12
LEFT_SIGN = 1
RIGHT_SIGN = -1
MIN_UPDATE_INTERVAL = 1 / 50  # 20ms


def value_to_duty_cycle(v):
    pulse_width_ms = 1.5 + (v * 0.5)  # 1ms to 2ms
    duty_cycle = int((pulse_width_ms / 20) * 65535)
    return min(max(duty_cycle, 3277), 6553)


class Antennas:
    def __init__(self):
        neutral_duty = value_to_duty_cycle(0)
        self.pwm_left = pwmio.PWMOut(LEFT_ANTENNA_PIN, frequency=50, duty_cycle=neutral_duty)
        self.pwm_right = pwmio.PWMOut(RIGHT_ANTENNA_PIN, frequency=50, duty_cycle=neutral_duty)

        # --- MEMÓRIA PARA SUAVIZAÇÃO (Filtro Passa-Baixa) ---
        # Guarda a posição atual real do servo
        self.current_left = 0.0
        self.current_right = 0.0
        
        # Fator de suavização: de 0.01 (muito lento) a 1.0 (imediato/tremedeira)
        # 0.1 significa que a antena percorre 10% da distância desejada a cada frame.
        self.smoothing = 0.1 
        # ----------------------------------------------------

    def set_position_left(self, target_position):
        # Interpolação: O valor atual se aproxima do alvo suavemente
        self.current_left += self.smoothing * (target_position - self.current_left)
        self.set_position(self.pwm_left, self.current_left, LEFT_SIGN)

    def set_position_right(self, target_position):
        # Interpolação: O valor atual se aproxima do alvo suavemente
        self.current_right += self.smoothing * (target_position - self.current_right)
        self.set_position(self.pwm_right, self.current_right, RIGHT_SIGN)

    def set_position(self, pwm, value, sign=1):
        if -1 <= value <= 1:
            duty_cycle = value_to_duty_cycle(value * sign)
            pwm.duty_cycle = duty_cycle
        else:
            print("Invalid input! Enter a value between -1 and 1.")

    def stop(self):
        time.sleep(MIN_UPDATE_INTERVAL)
        self.set_position_left(0)
        self.set_position_right(0)
        time.sleep(MIN_UPDATE_INTERVAL)
        self.pwm_left.deinit()
        self.pwm_right.deinit()


if __name__ == "__main__":
    antennas = Antennas()

    try:
        start_time = time.monotonic()
        current_time = start_time

        while current_time - start_time < 5:
            value = math.sin(2 * math.pi * 1 * current_time)
            antennas.set_position_left(value)
            antennas.set_position_right(value)
            time.sleep(MIN_UPDATE_INTERVAL)
            current_time = time.monotonic()

    finally:
        antennas.stop()