About the Project
The Open Duck Mini Runtime is the control engine for bipedal robots based on the Open Duck platform. This repository provides a robust Python-based execution environment, integrating artificial intelligence (neural networks), hardware control, and peripherals like antennas and sound systems.

🚀 Key Features
ONNX Inference: Real-time execution of Reinforcement Learning (RL) models for stability and walking gaits.

Hardware Interface (HWI): Direct communication with Feetech servos via optimized serial protocols.

Polynomial Kinematics: Smooth reference trajectory generation for natural movement.

Peripheral Support: Integrated management for servos (antennas), LED matrix (eyes), and sound effects via Pygame.

Smoothing Filters: Low-Pass Filter implementation to eliminate motor jitter.

🎮 Gamepad Operation
The robot is operated using a standard Xbox/PS4 controller:

Left Stick: Movement (Forward, Backward, Lateral).

Right Stick: Rotation and head orientation.

Triggers (LT/RT): Manual and smooth control of the antennas.

🛠️ Interactive Calibration Mode
This runtime includes a built-in terminal calibration tool:

Run the script and press the A Button (Xbox) to enter calibration mode.

Use the D-Pad to navigate between joints and adjust offsets (mechanical zeros).

Press B to permanently save settings to the duck_config.json file.

_____

How to calibrate while running the duck :

1. Entering Calibration Mode
To adjust the joints, the robot must be in "pause mode," where the motors relax to allow offset readings:

Activation: While the script is running, press the A button (Xbox) or X (PS4).

What happens: The robot stops walking, and the motor torque is reduced.

On Screen: The terminal will display: === MODE: CALIBRATION ACTIVE ===.

2. Adjustment Controls
In calibration mode, you use the D-Pad on the controller:

D-Pad Up/Down: Navigate through the joints (hip, knee, ankle, etc.).

D-Pad Left/Right: Increase or decrease the offset value for the selected joint.

B button (Xbox) or O (PS4): Permanently saves the changes to the duck_config.json file.

3. On-Screen Information
The terminal will update in real-time to guide you:

Selected Joint: Will be highlighted, e.g., CURRENT ADJUSTMENT: left_hip_pitch.

Offset Value: You will see the number changing as you press the arrows, e.g., Value: -0.05.

Save Confirmation: When pressing the save button, it will show: [OK] Settings saved to duck_config.json.

_________

Português (Brasil)
Sobre o Projeto
O Open Duck Mini Runtime é o "cérebro" de controle para robôs bípedes baseados na plataforma Open Duck. Este repositório foca em fornecer um ambiente de execução (runtime) robusto em Python, capaz de integrar inteligência artificial (redes neurais), controle via hardware e periféricos como antenas e sons.

🚀 Funcionalidades Principais
Inferência ONNX: Execução em tempo real de modelos de Reinforcement Learning (RL) para estabilidade e caminhada.

Controle de Hardware (HWI): Interface direta com servos Feetech através de comunicação serial otimizada.

Cinemática Polinomial: Geração de trajetórias de referência suaves para movimentos naturais.

Suporte a Periféricos: Gerenciamento integrado de antenas (servos), matriz de LEDs para olhos e efeitos sonoros via Pygame.

Filtros de Suavização: Implementação de filtros passa-baixa (Low-Pass Filter) para eliminar jitter nos motores.

🎮 Operação pelo Controle
O robô é operado nativamente através de um controle de Xbox/PS4:

Analógico Esquerdo: Movimentação (Frente, Trás, Lado).

Analógico Direito: Rotação e orientação da cabeça.

Gatilhos (LT/RT): Controle manual e suave das antenas.

🛠️ Modo de Calibração Interativo
Este runtime possui uma ferramenta de calibração integrada diretamente no terminal:

Inicie o script e pressione o Botão A (Xbox) para entrar no modo de calibração.

Use o D-Pad para navegar entre as juntas e ajustar os offsets (zeros mecânicos).

Pressione B para salvar as configurações permanentemente no arquivo duck_config.json.

______

1. Como Entrar no Modo de Calibração
Para ajustar as juntas, o robô precisa estar em "modo de pausa", onde os motores relaxam e permitem a leitura dos offsets:

Ativação: Com o código rodando, pressione o botão A (Xbox) ou X (PS4).

O que acontece: O robô para de tentar caminhar e o torque dos motores é aliviado.

No Terminal: A tela mostrará a mensagem: === MODO: CALIBRAÇÃO ATIVA ===.

2. Controles de Ajuste
No modo de calibração, você usa o D-Pad (as setinhas) do controle:

D-Pad Cima/Baixo: Navega entre as juntas (quadril, joelho, tornozelo, etc.).

D-Pad Esquerda/Direita: Aumenta ou diminui o valor de offset da junta selecionada.

Botão B (Xbox) ou O (PS4): Salva as alterações permanentemente no arquivo duck_config.json.

3. Informações na Tela
O terminal atualizará em tempo real para te guiar:

Junta Selecionada: Aparecerá em destaque, ex: AJUSTE ATUAL: left_hip_pitch.

Valor do Offset: Você verá o número mudando conforme aperta as setas, ex: Valor: -0.05.

Confirmação de Salve: Ao pressionar o botão de salvar, aparecerá: [OK] Configurações salvas em duck_config.json.
