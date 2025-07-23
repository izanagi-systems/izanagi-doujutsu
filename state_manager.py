from collections import deque
import time
from config import ESTADOS, PERFIL_CAIXA, TAMANHO_BUFFER_ESTABILIZACAO, TEMPO_LIMITE_CAIXA_AUSENTE

class StateManager:
    """
    Gerencia o estado do sistema, a lógica de transição e as regras de negócio.
    """
    def __init__(self):
        """
        Inicializa a máquina de estados e as variáveis de controle.
        """
        # --- Máquina de Estados e Variáveis de Controle ---
        self.status_sistema = ESTADOS['AGUARDANDO_CAIXA']
        self.estado_anterior = None
        self.camada_atual = 1
        self.contagens_por_camada = {i: 0 for i in range(1, PERFIL_CAIXA['total_camadas'] + 1)}
        self.contagem_estabilizada = 0

        # --- Buffers para Estabilização de Detecção ---
        self.buffer_roi = deque(maxlen=TAMANHO_BUFFER_ESTABILIZACAO)
        self.buffer_contagem_itens = deque(maxlen=TAMANHO_BUFFER_ESTABILIZACAO)
        self.buffer_divisor_presente = deque(maxlen=TAMANHO_BUFFER_ESTABILIZACAO)

        # --- Memória para Caixa Ausente ---
        self.caixa_ausente_desde = None
        self.ultima_roi_conhecida = None

    def atualizar_estado(self, roi, itens_na_roi, divisores_na_roi):
        """
        O coração da máquina de estados. Processa as detecções atuais
        e decide se deve mudar o estado do sistema.
        """
        # 1. Atualizar buffers com as detecções do frame atual
        self.buffer_roi.append(1 if roi else 0)
        self.buffer_contagem_itens.append(len(itens_na_roi))
        self.buffer_divisor_presente.append(1 if divisores_na_roi else 0)

        # 2. Obter valores estabilizados (só continua se os buffers estiverem cheios)
        if len(self.buffer_roi) < TAMANHO_BUFFER_ESTABILIZACAO:
            return # Aguardando buffers encherem

        # A ROI é considerada estável se estiver presente na maioria dos frames do buffer.
        roi_estavel = sum(self.buffer_roi) > (TAMANHO_BUFFER_ESTABILIZACAO / 2)
        # Para contagem, usamos a moda (valor mais comum) para robustez
        self.contagem_estabilizada = max(set(self.buffer_contagem_itens), key=self.buffer_contagem_itens.count)
        divisor_estavel = sum(self.buffer_divisor_presente) > 0 # Presente se detectado em pelo menos 1 frame do buffer

        # --- LÓGICA DA MÁQUINA DE ESTADOS --- 
        estado_atual = self.status_sistema

        if estado_atual == ESTADOS['AGUARDANDO_CAIXA']:
            if roi_estavel:
                print("[STATUS] Caixa detectada com estabilidade. Mudando para CONTANDO_ITENS.")
                self._transitar_para(ESTADOS['CONTANDO_ITENS'])
        
        elif estado_atual == ESTADOS['CONTANDO_ITENS']:
            if not roi_estavel:
                print("[STATUS] Caixa ausente durante a contagem. Iniciando temporizador.")
                self._transitar_para(ESTADOS['CAIXA_AUSENTE'])
                self.caixa_ausente_desde = time.time()
                # self.ultima_roi_conhecida = ... (lógica para salvar a posição exata pode ser adicionada aqui)
                return

            # Verifica se a camada está completa
            if self.contagem_estabilizada >= PERFIL_CAIXA['itens_esperados']:
                print(f"[STATUS] Camada {self.camada_atual} completa. Aguardando divisor.")
                self._transitar_para(ESTADOS['AGUARDANDO_DIVISOR'])

        elif estado_atual == ESTADOS['AGUARDANDO_DIVISOR']:
            if not roi_estavel:
                print("[STATUS] Caixa ausente enquanto aguardava divisor. Pausando.")
                self._transitar_para(ESTADOS['CAIXA_AUSENTE'])
                self.caixa_ausente_desde = time.time()
                return

            if divisor_estavel:
                # Salva a contagem da camada que acabamos de fechar
                self.contagens_por_camada[self.camada_atual] = self.contagem_estabilizada
                print(f"[INFO] Divisor detectado. Camada {self.camada_atual} finalizada com {self.contagem_estabilizada} itens.")

                if self.camada_atual >= PERFIL_CAIXA['total_camadas']:
                    print("[STATUS] Todas as camadas foram preenchidas. Caixa completa!")
                    self._transitar_para(ESTADOS['CAIXA_COMPLETA'])
                else:
                    self.camada_atual += 1
                    print(f"[STATUS] Iniciando contagem para a camada {self.camada_atual}.")
                    self._transitar_para(ESTADOS['CONTANDO_ITENS'])

        elif estado_atual == ESTADOS['CAIXA_COMPLETA']:
            # O sistema aguarda a caixa ser removida para reiniciar o ciclo.
            if not roi_estavel:
                print("[INFO] Caixa completa removida. Reiniciando o ciclo para a próxima caixa.")
                self._resetar_sistema()

        elif estado_atual == ESTADOS['CAIXA_AUSENTE']:
            if roi_estavel:
                print("[STATUS] Caixa reapareceu. Retomando contagem.")
                self._transitar_para(self.estado_anterior) # Volta para o estado que estava antes da ausência
                self.caixa_ausente_desde = None
            
            elif self.caixa_ausente_desde and (time.time() - self.caixa_ausente_desde > TEMPO_LIMITE_CAIXA_AUSENTE):
                print("[ALERTA] Tempo limite de ausência da caixa excedido. Resetando o sistema.")
                self._resetar_sistema()

    def _resetar_sistema(self):
        """Reseta o estado do sistema para o inicial."""
        print("[INFO] Sistema resetado.")
        self._transitar_para(ESTADOS['AGUARDANDO_CAIXA'])
        self.camada_atual = 1
        self.contagens_por_camada = {i: 0 for i in range(1, PERFIL_CAIXA['total_camadas'] + 1)}
        self.contagem_estabilizada = 0
        self.caixa_ausente_desde = None
        self.ultima_roi_conhecida = None
        # Limpa os buffers para começar do zero
        self.buffer_roi.clear()
        self.buffer_contagem_itens.clear()
        self.buffer_divisor_presente.clear()

    def _transitar_para(self, novo_estado):
        """Método auxiliar para registrar a mudança de estado."""
        print(f"[FSM] Transição de {self.status_sistema} para {novo_estado}")
        self.estado_anterior = self.status_sistema
        self.status_sistema = novo_estado

    def get_status_visual(self):
        """
        Retorna uma representação textual do estado atual para a visualização.
        (Esta função será aprimorada)
        """
        return {
            'status_texto': self.status_sistema,
            'contagem': self.contagem_estabilizada,
            'camada': self.camada_atual
        }
