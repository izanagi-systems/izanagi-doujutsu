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
    'CAIXA_COMPLETA': 'Caixa Completa',
    'VERIFICANDO_CAMADA': 'Verificando Camada',
    'ERRO_DIVISOR_PRECOCE': 'Erro: Divisor Detectado Precocemente',
    'CAIXA_AUSENTE': 'Alerta: Caixa Ausente',
    'ALERTA_DIVISOR_AUSENTE': 'Alerta: Divisor da Camada Ausente'
}

# --- Configurações dos Modelos YOLO ---
# Caminhos para os modelos treinados.
MODELOS = {
    'item_detector': 'modelos_producao/item_detector.pt',
    'roi_detector': 'modelos_producao/roi_detector.pt'
}
# Limite de confiança para as detecções do modelo.
CONFIDENCIA_LIMITE = 0.4
# Configurações específicas para detecção de divisores
CONFIDENCIA_DIVISOR = 0.3  # Confiança mais baixa para divisores
DEBUG_DIVISORES = False    # Ativar logs detalhados de divisores (desativado para reduzir spam)
DEBUG_DIVISORES_VERBOSE = False  # Logs muito detalhados apenas quando necessário

# --- Configurações de Estabilização e Memória ---
# Número de frames consecutivos para uma detecção ser considerada "estável".
TAMANHO_BUFFER_ESTABILIZACAO = 5
# Tempo em segundos que o sistema espera por uma caixa que sumiu antes de resetar
TEMPO_LIMITE_CAIXA_AUSENTE = 10
# Tempo em segundos para resetar o processo se a caixa não retornar.
TEMPO_LIMITE_AUSENCIA = 30.0

# --- Configurações para Prevenção de Falsos Positivos ---
# Distância mínima (em pixels) para considerar um item como "novo" entre camadas
DISTANCIA_MINIMA_ITEM_NOVO = 50
# Percentual mínimo de itens que devem ser "novos" para validar uma nova camada
PERCENTUAL_ITENS_NOVOS_MINIMO = 0.7  # 70% dos itens devem ser novos

# --- Configurações Avançadas da Camada 2 ---
# Lógica inteligente para transição e manutenção da camada 2
ITENS_MINIMOS_CAMADA_2_ESTABELECIDA = 5  # A partir de 5 itens, considera camada 2 estabelecida
TEMPO_CARENCIA_DIVISOR_AUSENTE = 3.0  # Segundos de carência antes de alertar divisor ausente
TEMPO_CARENCIA_CONTAGEM_BAIXA = 2.0   # Segundos de carência para contagem baixa

# --- Configurações de Detecção de Saltos Híbrida ---
# Sistema para detectar e validar saltos anômalos na contagem
SALTO_SUSPEITO_MINIMO = 3      # Salto mínimo considerado suspeito (>3 itens)
TEMPO_MAXIMO_SALTO = 2.0       # Tempo máximo para considerar salto suspeito (segundos)
TEMPO_CARENCIA_SALTO = 3.0     # Tempo de carência para validar salto (segundos)
PERCENTUAL_ITENS_NOVOS_SALTO = 0.7  # 70% dos itens devem ser novos para aceitar salto

# --- Configurações de Tolerância a Oclusões ---
# Sistema para tolerar oclusões naturais durante preenchimento
TOLERANCIA_OCLUSAO_CAMADA_2 = True    # Ativar tolerância a oclusões na camada 2
SALTO_OCLUSAO_MAXIMO = 2              # Salto máximo considerado oclusão natural
TEMPO_CARENCIA_PERDA_CAIXA = 3.0      # Carência antes de resetar por perda de caixa (segundos)

# --- Configurações de Memória Espacial ---
# Sistema para evitar recontagem de itens entre camadas
USAR_MEMORIA_ESPACIAL = True

# --- Constantes de Desenho e UI ---
# Cores usadas para desenhar os elementos na tela (formato BGR).
CORES = {
    'roi': (255, 0, 255),          # Rosa para ROI
    'item_ok': (0, 255, 0),        # Verde para itens contados
    'divisor': (0, 255, 255),      # Ciano para divisor
    'texto_status': (255, 100, 0),   # Azul vivo para status (bem visível)
    'texto_contagem': (255, 150, 0), # Azul vivo claro para contagem
    'alerta': (0, 0, 255)          # Vermelho para alertas e erros
}

# Fonte e espessura das linhas para os textos e retângulos.
FONTE = cv2.FONT_HERSHEY_SIMPLEX
ESPESSURA_LINHA = 2