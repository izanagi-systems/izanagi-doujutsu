import cv2
from collections import deque

# --- Perfil da Caixa (Configuração do Processo) ---
# Define as regras para uma caixa ser considerada "completa".
PERFIL_CAIXA = {
    'itens_esperados': 12,
    'total_camadas': 2
}

# --- Configurações da Máquina de Estados ---
ESTADOS = {
    'AGUARDANDO_CAIXA': 'AGUARDANDO_CAIXA',
    'CONTANDO_ITENS': 'CONTANDO_ITENS',
    'AGUARDANDO_DIVISOR': 'AGUARDANDO_DIVISOR',
    'CAIXA_COMPLETA': 'CAIXA_COMPLETA',
    'ERRO_DIVISOR_PRECOCE': 'ERRO_DIVISOR_PRECOCE',
    'CAIXA_AUSENTE': 'CAIXA_AUSENTE',
    'ALERTA_DIVISOR_AUSENTE': 'ALERTA_DIVISOR_AUSENTE'
}

# --- Configurações dos Modelos YOLO ---
# Caminhos para os modelos treinados.
MODELOS = {
    'item_detector': 'modelos_producao/item_detector.pt',
    'roi_detector': 'modelos_producao/roi_detector.pt'
}
# Limite de confiança para as detecções do modelo.
CONFIDENCIA_LIMITE = 0.4

# --- Configurações de Estabilização e Memória ---
# Número de frames consecutivos para uma detecção ser considerada "estável".
TAMANHO_BUFFER_ESTABILIZACAO = 5
# Tempo em segundos que o sistema espera por uma caixa que sumiu antes de resetar
TEMPO_LIMITE_CAIXA_AUSENTE = 10
# Tempo em segundos para resetar o processo se a caixa não retornar.
TEMPO_LIMITE_AUSENCIA = 30.0

# --- Constantes de Desenho e UI ---
# Cores usadas para desenhar os elementos na tela (formato BGR).
CORES = {
    'roi': (255, 0, 255),          # Rosa para ROI
    'item_ok': (0, 255, 0),        # Verde para itens contados
    'divisor': (0, 255, 255),      # Ciano para divisor
    'texto_status': (0, 0, 0),     # Preto para status
    'texto_contagem': (255, 0, 0),  # Azul para contagem
    'alerta': (0, 0, 255)          # Vermelho para alertas e erros
}

# Fonte e espessura das linhas para os textos e retângulos.
FONTE = cv2.FONT_HERSHEY_SIMPLEX
ESPESSURA_LINHA = 2