import adafruit_bno055
import board
import busio
import numpy as np
import os
import pickle
import sys  # Importado para sair limpo

from queue import Queue
from threading import Thread
import time

class Imu:
    def __init__(
        self, sampling_freq, user_pitch_bias=0, calibrate=False, upside_down=True
    ):
        self.sampling_freq = sampling_freq
        self.calibrate = calibrate

        # Configuração I2C
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.imu = adafruit_bno055.BNO055_I2C(i2c, address=0x29)
        except Exception as e:
            print(f"Erro ao conectar IMU: {e}")
            # Fallback para debug sem sensor, se necessário
            return

        self.imu.mode = adafruit_bno055.NDOF_MODE

        # Configuração de Eixos
        if upside_down:
            self.imu.axis_remap = (
                adafruit_bno055.AXIS_REMAP_Y,
                adafruit_bno055.AXIS_REMAP_X,
                adafruit_bno055.AXIS_REMAP_Z,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
            )
        else:
            self.imu.axis_remap = (
                adafruit_bno055.AXIS_REMAP_Y,
                adafruit_bno055.AXIS_REMAP_X,
                adafruit_bno055.AXIS_REMAP_Z,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_POSITIVE,
                adafruit_bno055.AXIS_REMAP_POSITIVE,
            )

        # --- LÓGICA DE CALIBRAÇÃO MELHORADA ---
        if self.calibrate:
            print("--------------------------------------")
            print("MODO DE CALIBRAÇÃO IMU INICIADO")
            print("Movimente o robô para calibrar.")
            print("OBJETIVO: Gyro=3, Accel=2 ou 3.")
            print("PRESSIONE CTRL+C PARA SALVAR E SAIR.")
            print("--------------------------------------")
            
            self.imu.mode = adafruit_bno055.NDOF_MODE
            
            try:
                while True:
                    sys_stat, gyro_stat, accel_stat, mag_stat = self.imu.calibration_status
                    
                    # Mostra status em tempo real
                    status_str = f"Sys: {sys_stat} | Gyro: {gyro_stat} | Accel: {accel_stat} | Mag: {mag_stat}"
                    print(status_str, end="\r") # \r para atualizar na mesma linha
                    
                    # Se tudo estiver perfeito, avisa (mas não sai sozinho, deixa o usuário decidir)
                    if sys_stat == 3 and gyro_stat == 3 and accel_stat == 3 and mag_stat == 3:
                        print("\nCALIBRAÇÃO PERFEITA ATINGIDA! (Pressione Ctrl+C para salvar)")
                    
                    time.sleep(0.1)

            except KeyboardInterrupt:
                print("\n\nCapturado Ctrl+C. Salvando calibração atual...")
                
                # Coleta os offsets atuais (mesmo que não perfeitos)
                offsets_accelerometer = self.imu.offsets_accelerometer
                offsets_gyroscope = self.imu.offsets_gyroscope
                offsets_magnetometer = self.imu.offsets_magnetometer

                imu_calib_data = {
                    "offsets_accelerometer": offsets_accelerometer,
                    "offsets_gyroscope": offsets_gyroscope,
                    "offsets_magnetometer": offsets_magnetometer,
                }
                
                print("Valores capturados:")
                for k, v in imu_calib_data.items():
                    print(f"{k}: {v}")

                pickle.dump(imu_calib_data, open("imu_calib_data.pkl", "wb"))
                print("\nARQUIVO 'imu_calib_data.pkl' SALVO COM SUCESSO!")
                exit()
        # --------------------------------------

        # Carrega calibração se existir
        if os.path.exists("imu_calib_data.pkl"):
            try:
                imu_calib_data = pickle.load(open("imu_calib_data.pkl", "rb"))
                self.imu.mode = adafruit_bno055.CONFIG_MODE
                time.sleep(0.1)
                self.imu.offsets_accelerometer = imu_calib_data["offsets_accelerometer"]
                self.imu.offsets_gyroscope = imu_calib_data["offsets_gyroscope"]
                self.imu.offsets_magnetometer = imu_calib_data["offsets_magnetometer"]
                self.imu.mode = adafruit_bno055.NDOF_MODE
                time.sleep(0.1)
                print("Calibração carregada do arquivo.")
            except Exception as e:
                print(f"Erro ao carregar calibração: {e}")
        else:
            print("imu_calib_data.pkl not found")
            print("Imu is running uncalibrated")

        self.x_offset = 0
        self.last_imu_data = {
            "gyro": [0, 0, 0],
            "accelero": [0, 0, 0],
        }
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def imu_worker(self):
        while True:
            s = time.time()
            try:
                gyro = np.array(self.imu.gyro).copy()
                accelero = np.array(self.imu.acceleration).copy()
            except Exception as e:
                # print("[IMU]:", e) # Comentado para não poluir o terminal se der erro pontual
                continue

            if gyro is None or accelero is None:
                continue

            # Verificação se arrays estão vazios ou None
            if gyro.size == 0 or accelero.size == 0:
                continue

            accelero[0] -= self.x_offset

            data = {
                "gyro": gyro,
                "accelero": accelero,
            }

            self.imu_queue.put(data)
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def get_data(self):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data


if __name__ == "__main__":
    # Para rodar a calibração, mude calibrate=True abaixo
    # Ou rode o script e depois mude no código principal
    
    # Exemplo de uso normal (leitura):
    # imu = Imu(50, upside_down=False, calibrate=False)
    
    # Exemplo PARA CALIBRAR (descomente a linha abaixo e rode este script):
    imu = Imu(50, upside_down=False, calibrate=True)

    while True:
        data = imu.get_data()
        if data:
            print("gyro", np.around(data["gyro"], 3))
            print("accelero", np.around(data["accelero"], 3))
            print("---")
        time.sleep(1 / 25)
