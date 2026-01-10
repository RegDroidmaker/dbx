import time
import pickle
import os
import numpy as np
import pygame  # Importante para controlar o audio
import random  # Para escolher o som aleatório
import json    # Para salvar o JSON

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
from mini_bdx_runtime.duck_config import DuckConfig

HOME_DIR = os.path.expanduser("~")

class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        duck_config_path: str = os.path.join(HOME_DIR, "dbx", "duck_config.json"),
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=0.25,
        commands=False,
        pitch_bias=0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
    ):
        # --- Caminhos absolutos para PKL e Assets ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(self.script_dir, "polynomial_coefficients.pkl")
        assets_path = os.path.join(self.script_dir, "../mini_bdx_runtime/assets/")
        # --------------------------------------------------------

        self.duck_config_path = duck_config_path
        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        # --- SISTEMA DE CALIBRAÇÃO DE OFFSETS ---
        self.joint_names = list(self.duck_config.joints_offset.keys())
        self.selected_joint_index = 0
        # ----------------------------------------

        self.commands = commands
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14
        self.max_motor_velocity = 5.24  # rad/s

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        self.hwi = HWI(self.duck_config, serial_port)

        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = FeetContacts()

        # Scales
        self.action_scale = action_scale

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = list(self.hwi.init_pos.values())

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused

        self.command_freq = 20  # hz
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)

        self.PRM = PolyReferenceMotion(pkl_path)
        
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        # Optional expression features
        if self.duck_config.eyes:
            self.eyes = Eyes()

        # --- CONTROLE DO PROJETOR ---
        self.projector_active = False 
        if self.duck_config.projector:
            self.projector = Projector()

        # --- CONTROLE DE SOM ---
        self.last_random_sound = None 
        if self.duck_config.speaker:
            self.sounds = Sounds(
                volume=1.0, sound_directory=assets_path
            )

        if self.duck_config.antennas:
            self.antennas = Antennas()

    def save_offsets_to_json(self):
        """Salva os offsets atuais no arquivo JSON para persistencia"""
        try:
            self.duck_config.json_config["joints_offsets"] = self.duck_config.joints_offset
            with open(self.duck_config_path, 'w') as f:
                json.dump(self.duck_config.json_config, f, indent=4)
            print("Configuracao salva com sucesso!")
        except Exception as e:
            print(f"Erro ao salvar JSON: {e}")

    def get_obs(self):

        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        cmds = self.last_commands

        feet_contacts = self.feet_contacts.get()

        obs = np.concatenate(
            [
                imu_data["gyro"],
                imu_data["accelero"],
                cmds,
                dof_pos - self.init_pos,
                dof_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                feet_contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [8, 8, 8, 8]

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def get_phase_frequency_factor(self, x_velocity):

        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # Perform linear interpolation
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )

        return freq

    def run(self):
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    
                    # --- LÓGICA CONDICIONAL: PAUSADO ou NORMAL ---
                    if not self.paused:
                        # === MODO NORMAL ===
                        if self.buttons.dpad_up.triggered:
                            self.phase_frequency_factor_offset += 0.05
                            print(f"Speed Offset: {round(self.phase_frequency_factor_offset, 3)}")

                        if self.buttons.dpad_down.triggered:
                            self.phase_frequency_factor_offset -= 0.05
                            print(f"Speed Offset: {round(self.phase_frequency_factor_offset, 3)}")
                    else:
                        # === MODO PAUSE (CALIBRAÇÃO) ===
                        if self.buttons.LB.triggered:
                            self.selected_joint_index = (self.selected_joint_index - 1) % len(self.joint_names)
                            motor_name = self.joint_names[self.selected_joint_index]
                            current_val = self.duck_config.joints_offset[motor_name]
                            print(f"SELECIONADO: {motor_name} | Offset Atual: {current_val}")

                        if self.buttons.RB.triggered:
                            self.selected_joint_index = (self.selected_joint_index + 1) % len(self.joint_names)
                            motor_name = self.joint_names[self.selected_joint_index]
                            current_val = self.duck_config.joints_offset[motor_name]
                            print(f"SELECIONADO: {motor_name} | Offset Atual: {current_val}")

                        if self.buttons.dpad_up.triggered:
                            motor_name = self.joint_names[self.selected_joint_index]
                            self.duck_config.joints_offset[motor_name] += 0.01
                            self.duck_config.joints_offset[motor_name] = round(self.duck_config.joints_offset[motor_name], 3)
                            print(f"AJUSTE: {motor_name} -> {self.duck_config.joints_offset[motor_name]}")
                            self.save_offsets_to_json()

                        if self.buttons.dpad_down.triggered:
                            motor_name = self.joint_names[self.selected_joint_index]
                            self.duck_config.joints_offset[motor_name] -= 0.01
                            self.duck_config.joints_offset[motor_name] = round(self.duck_config.joints_offset[motor_name], 3)
                            print(f"AJUSTE: {motor_name} -> {self.duck_config.joints_offset[motor_name]}")
                            self.save_offsets_to_json()
                    # ----------------------------------------------

                    if self.buttons.LB.is_pressed and not self.paused:
                        self.phase_frequency_factor = 1.3
                    else:
                        self.phase_frequency_factor = 1.0

                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()
                            self.projector_active = not self.projector_active
                            
                            if self.duck_config.speaker:
                                if self.projector_active:
                                    try:
                                        self.sounds.play("projector.wav")
                                    except AttributeError:
                                        sound_p = os.path.join(self.script_dir, "../mini_bdx_runtime/assets/projector.wav")
                                        pygame.mixer.Sound(sound_p).play()
                                else:
                                    pygame.mixer.stop()

                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            if pygame.mixer.get_busy():
                                pygame.mixer.stop()
                            else:
                                try:
                                    base_assets = os.path.join(self.script_dir, "../mini_bdx_runtime/assets/")
                                    random_dir = os.path.join(base_assets, "random")
                                    if not os.path.exists(random_dir):
                                        random_dir = base_assets

                                    files = [f for f in os.listdir(random_dir) if f.endswith('.wav') or f.endswith('.mp3')]
                                    if files:
                                        available_sounds = [f for f in files if f != self.last_random_sound]
                                        if not available_sounds:
                                            available_sounds = files

                                        chosen_sound = random.choice(available_sounds)
                                        self.last_random_sound = chosen_sound
                                        full_path = os.path.join(random_dir, chosen_sound)
                                        print(f"Tocando: {chosen_sound}")
                                        pygame.mixer.Sound(full_path).play()
                                except Exception as e:
                                    print(f"Erro ao tocar som: {e}")

                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        if self.paused:
                            print("\n=== MODO PAUSE: CALIBRAÇÃO ATIVADA ===")
                            motor_name = self.joint_names[self.selected_joint_index]
                            print(f"Motor Atual: {motor_name} ({self.duck_config.joints_offset[motor_name]})")
                        else:
                            print("=== MODO RUN: CALIBRAÇÃO SALVA E APLICADA ===")
                            # --- RESET SUAVE AO SAIR DO PAUSE ---
                            # Reseta os buffers de ação para evitar "pulo"
                            self.last_action = np.zeros(self.num_dofs)
                            self.last_last_action = np.zeros(self.num_dofs)
                            self.last_last_last_action = np.zeros(self.num_dofs)
                            
                            # Atualiza a init_pos com os novos offsets que você acabou de configurar
                            # O HWI já leu os novos offsets do self.duck_config automaticamente? 
                            # Não, precisamos forçar uma "releitura" ou apenas confiar que 
                            # o init_pos + action vai funcionar porque o init_pos é fixo 
                            # mas o HWI aplica o offset internamente?
                            #
                            # Na verdade, a classe HWI lê os offsets no __init__.
                            # Se mudarmos os offsets no duck_config aqui fora, precisamos avisar o HWI
                            # ou recalcular o init_pos.
                            #
                            # Vamos simplificar: O init_pos é a posição "zero" dos motores. 
                            # Se mudamos o offset, o "zero" físico mudou. 
                            # Vamos atualizar o init_pos localmente para garantir.
                            self.init_pos = list(self.hwi.init_pos.values()) 
                            
                            # Reseta os alvos para a posição inicial atualizada
                            self.motor_targets = np.array(self.init_pos.copy())
                            self.prev_motor_targets = np.array(self.init_pos.copy())
                            # ------------------------------------

                if self.paused:
                    # --- MODO CALIBRAÇÃO (EM TEMPO REAL) ---
                    # Precisamos atualizar o offset dentro do HWI também, senão ele usa o velho.
                    # O HWI guarda os offsets em self.hwi.duck_config.joints_offset.
                    # Como passamos o objeto duck_config por referência, ao mudar aqui (self.duck_config),
                    # deve mudar lá também. Vamos garantir enviando o comando.
                    
                    action_dict = make_action_dict(
                        self.init_pos, list(self.hwi.joints.keys())
                    )
                    # O truque: set_position_all no HWI pega (target + offset).
                    # Se self.init_pos é 0 e offset mudou, o motor deve mover.
                    self.hwi.set_position_all(action_dict)
                    
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    continue

                self.imitation_i += 1 * (
                    self.phase_frequency_factor + self.phase_frequency_factor_offset
                )
                self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                self.imitation_phase = np.array(
                    [
                        np.cos(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                        np.sin(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                    ]
                )

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                action = self.policy.infer(obs)

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                self.motor_targets = self.init_pos + action * self.action_scale

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (
                        time.time() - start_t > 1
                    ):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets

                self.prev_motor_targets = self.motor_targets.copy()

                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )

                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            print("\nFinalizando componentes...")
            if self.duck_config.antennas:
                self.antennas.stop()
            if self.duck_config.eyes:
                self.eyes.stop()
            if self.duck_config.projector:
                self.projector.stop()
            self.feet_contacts.stop()
            
            if pygame.mixer.get_init():
                pygame.mixer.stop()

            print("Desligando motores (HWI turn_off)...")
            self.hwi.turn_off()
            time.sleep(1)

        if self.save_obs:
            pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
        print("TURNING OFF")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=os.path.join(HOME_DIR, "dbx", "duck_config.json"),
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
    )
    print("Done instantiating RLWalk")
    rl_walk.run()
