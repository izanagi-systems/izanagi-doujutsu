from collections import deque
import time
import math
from config import (
    ESTADOS, PERFIL_CAIXA, TAMANHO_BUFFER_ESTABILIZACAO, TEMPO_LIMITE_CAIXA_AUSENTE,
    USAR_MEMORIA_ESPACIAL, DISTANCIA_MINIMA_ITEM_NOVO, PERCENTUAL_ITENS_NOVOS_MINIMO,
    ITENS_MINIMOS_CAMADA_2_ESTABELECIDA, TEMPO_CARENCIA_DIVISOR_AUSENTE, TEMPO_CARENCIA_CONTAGEM_BAIXA,
    SALTO_SUSPEITO_MINIMO, TEMPO_MAXIMO_SALTO, TEMPO_CARENCIA_SALTO, PERCENTUAL_ITENS_NOVOS_SALTO,
    TOLERANCIA_OCLUSAO_CAMADA_2, SALTO_OCLUSAO_MAXIMO, TEMPO_CARENCIA_PERDA_CAIXA,
    DEBUG_DIVISORES
)
from logger_config import get_siac_logger, SiacLogger

class StateManager:
    """
    Gerencia o estado do sistema, a lógica de transição e as regras de negócio.
    """
    def __init__(self):
        """
        Inicializa a máquina de estados e as variáveis de controle.
        """
        # Inicializar logger
        self.logger = get_siac_logger("STATE_MANAGER")
        
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
        
        # --- Sistema de Memória Espacial para Prevenção de Falsos Positivos ---
        self.posicoes_itens_por_camada = {}  # Armazena posições dos itens de cada camada
        self.usar_memoria_espacial = USAR_MEMORIA_ESPACIAL
        
        # --- Controles Especiais ---
        # Controle de timing para divisores (evita contagem prematura)
        self.divisor_detectado_frames = 0
        self.frames_minimos_divisor = 3  # Aguardar 3 frames após detectar divisor
        
        # Flag para detectar início com caixa já cheia
        self.primeira_deteccao = True
        
        # --- Controles Avançados da Camada 2 ---
        self.camada_2_estabelecida = False  # Se a camada 2 já foi estabelecida (5+ itens)
        self.tempo_ultimo_divisor_ausente = None  # Para carência de alertas
        self.tempo_ultima_contagem_baixa = None   # Para carência de contagem baixa
        self.divisor_cobrindo_itens = False  # Se o divisor já cobriu todos os itens (contagem = 0)
        
        # --- Controles de Detecção de Saltos Híbrida ---
        self.contagem_anterior_camada_2 = 0  # Última contagem conhecida da camada 2
        self.tempo_ultima_contagem_camada_2 = None  # Timestamp da última contagem
        self.salto_suspeito_detectado = False  # Se há um salto suspeito em validação
        self.tempo_inicio_salto_suspeito = None  # Quando o salto suspeito foi detectado
        self.itens_salto_suspeito = []  # Itens do salto suspeito para validação
        
        # --- Controles de Carência para Perda de Caixa ---
        self.tempo_perda_caixa = None  # Quando a caixa foi perdida
        self.estado_antes_perda_caixa = None  # Estado anterior à perda da caixa
        
        # --- Controles de Debounce para Alertas ---
        self.ultimo_alerta_tempo = None  # Timestamp do último alerta
        self.ultimo_alerta_tipo = None   # Tipo do último alerta emitido
        
        self.logger.info("StateManager inicializado")
        self.logger.info(f"Configuração: {PERFIL_CAIXA['total_camadas']} camadas, {PERFIL_CAIXA['itens_esperados']} itens por camada")
        self.logger.info(f"Memória espacial: {'Ativada' if self.usar_memoria_espacial else 'Desativada'}")

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
                self.logger.info("Caixa detectada com estabilidade")
                self._transitar_para(ESTADOS['CONTANDO_ITENS'], "ROI estável detectada")
        
        elif estado_atual == ESTADOS['CONTANDO_ITENS']:
            # Defesa Nível 1: Se a caixa sumir, aplicar lógica inteligente
            if not roi_estavel:
                tempo_atual = time.time()
                
                # ALERTA IMEDIATO para caixa incompleta (sem carência)
                if self.contagem_estabilizada > 0 and self.contagem_estabilizada < PERFIL_CAIXA['itens_por_camada']:
                    if self._pode_alertar("caixa_incompleta", 3.0):
                        self.logger.error(f"🚨 ALERTA IMEDIATO: Caixa removida INCOMPLETA! Camada {self.camada_atual}: {self.contagem_estabilizada}/{PERFIL_CAIXA['itens_por_camada']} itens")
                        self.logger.error(f"⚠️  Caixa retirada com contagem em andamento - SEM carência")
                    
                    # Ir direto para CAIXA_AUSENTE sem carência
                    self._transitar_para(ESTADOS['CAIXA_AUSENTE'], "ROI perdida - caixa incompleta")
                    self.caixa_ausente_desde = tempo_atual
                    self.tempo_perda_caixa = None
                    return
                
                # Carência normal para outros casos
                if self.tempo_perda_caixa is None:
                    # Primeira detecção de perda, iniciar carência
                    self.tempo_perda_caixa = tempo_atual
                    self.estado_antes_perda_caixa = self.status_sistema
                    self.logger.debug(f"Caixa perdida, iniciando carência de {TEMPO_CARENCIA_PERDA_CAIXA}s")
                    return
                
                tempo_carencia = tempo_atual - self.tempo_perda_caixa
                if tempo_carencia < TEMPO_CARENCIA_PERDA_CAIXA:
                    # Ainda em carência, aguardar
                    self.logger.debug(f"Carência perda caixa: {tempo_carencia:.1f}/{TEMPO_CARENCIA_PERDA_CAIXA}s")
                    return
                
                # Carência expirada para casos normais
                if self._pode_alertar("caixa_ausente", 3.0):
                    self.logger.warning(f"Caixa ausente por {tempo_carencia:.1f}s. Pausando contagem.")
                
                self._transitar_para(ESTADOS['CAIXA_AUSENTE'], "ROI perdida após carência")
                self.caixa_ausente_desde = tempo_atual
                self.tempo_perda_caixa = None
                return
            else:
                # Caixa presente, reset carência
                if self.tempo_perda_caixa is not None:
                    tempo_carencia = time.time() - self.tempo_perda_caixa
                    self.logger.debug(f"Caixa recuperada após {tempo_carencia:.1f}s de carência")
                    self.tempo_perda_caixa = None
                    self.estado_antes_perda_caixa = None

            # Lógica avançada para camada 2+
            if self.camada_atual > 1:
                # Primeiro, verificar se há salto suspeito na contagem
                contagem_atual = len(itens_na_roi)
                salto_validado = self._processar_deteccao_saltos_hibrida(contagem_atual, itens_na_roi)
                
                if not salto_validado:
                    # Salto suspeito detectado e não validado, pausar processamento
                    return
                
                self._processar_logica_camada_2(divisor_estavel, contagem_atual)
                if self.status_sistema == ESTADOS['ALERTA_DIVISOR_AUSENTE']:
                    return

            # Somente se as condições acima forem atendidas, prosseguimos para a lógica de conclusão.
            camada_completa = self.contagem_estabilizada >= PERFIL_CAIXA['itens_esperados']

            if camada_completa:
                SiacLogger.log_layer_completion(
                    self.logger, 
                    self.camada_atual, 
                    self.contagem_estabilizada, 
                    PERFIL_CAIXA['itens_esperados']
                )
                
                # Armazenar posições dos itens da camada completa
                if self.usar_memoria_espacial:
                    self.posicoes_itens_por_camada[self.camada_atual] = itens_na_roi.copy()
                    self.logger.info(f"Posições da camada {self.camada_atual} armazenadas: {len(itens_na_roi)} itens")
                
                # Lógica diferenciada por camada
                if self.camada_atual == 1:
                    # Camada 1: Aguardar divisor obrigatório
                    self._transitar_para(ESTADOS['AGUARDANDO_DIVISOR'], f"Camada {self.camada_atual} completa - aguardando divisor")
                elif self.camada_atual == PERFIL_CAIXA['total_camadas']:
                    # Última camada: Finalizar direto, sem procurar divisor
                    self.contagens_por_camada[self.camada_atual] = self.contagem_estabilizada
                    total_itens = sum(self.contagens_por_camada.values())
                    self.logger.info(f"Última camada completa! Finalizando caixa. Total: {total_itens} itens")
                    self._transitar_para(ESTADOS['CAIXA_COMPLETA'], "Todas as camadas completas")
                else:
                    # Camadas intermediárias: Verificar divisor
                    self._transitar_para(ESTADOS['VERIFICANDO_CAMADA'], f"Camada {self.camada_atual} completa")

        elif estado_atual == ESTADOS['VERIFICANDO_CAMADA']:
            # Neste estado, a contagem já está estabilizada no máximo.
            # A única tarefa é validar a presença do divisor para tomar a decisão final.

            # Se a caixa sumir durante a verificação, reseta o processo.
            if not roi_estavel:
                self.logger.warning("Caixa ausente durante a verificação")
                self._transitar_para(ESTADOS['AGUARDANDO_CAIXA'], "ROI perdida durante verificação")
                return

            # Controle de timing do divisor - aguardar alguns frames para estabilizar
            if len(divisores_na_roi) > 0:
                self.divisor_detectado_frames += 1
                if self.divisor_detectado_frames < self.frames_minimos_divisor:
                    self.logger.debug(f"Divisor detectado, aguardando estabilização ({self.divisor_detectado_frames}/{self.frames_minimos_divisor})")
                    return
            else:
                self.divisor_detectado_frames = 0

            if divisor_estavel:
                # SUCESSO: Contagem máxima e divisor presente.
                self.logger.info(f"Verificação da camada {self.camada_atual} bem-sucedida (Divisor presente)")
                
                # Armazenar posições dos itens da camada atual
                if self.usar_memoria_espacial:
                    self.posicoes_itens_por_camada[self.camada_atual] = itens_na_roi.copy()
                    self.logger.info(f"Posições da camada {self.camada_atual} armazenadas: {len(itens_na_roi)} itens")
                
                self.contagens_por_camada[self.camada_atual] = self.contagem_estabilizada

                if self.camada_atual < PERFIL_CAIXA['total_camadas']:
                    # Avança para a próxima camada
                    self.camada_atual += 1
                    self.logger.info(f"Iniciando contagem para a camada {self.camada_atual}")
                    self.buffer_contagem_itens.clear()  # Zera para a nova camada
                    self._transitar_para(ESTADOS['CONTANDO_ITENS'], f"Avançando para camada {self.camada_atual}")
                else:
                    # Finaliza a caixa
                    total_itens = sum(self.contagens_por_camada.values())
                    self.logger.info(f"Todas as camadas foram preenchidas. Caixa completa! Total: {total_itens} itens")
                    self._transitar_para(ESTADOS['CAIXA_COMPLETA'], "Todas as camadas completas")
            else:
                # FALHA: Contagem máxima mas SEM divisor. Verificar se é realmente falso positivo.
                
                # CASO ESPECIAL: Se é a primeira detecção e temos 12 itens, aceitar como camada 1 válida
                if self.primeira_deteccao and self.camada_atual == 1:
                    self.logger.info("Primeira detecção com caixa já cheia. Aceitando como camada 1 válida.")
                    self.primeira_deteccao = False
                    
                    # Armazenar posições e avançar para próxima camada
                    if self.usar_memoria_espacial:
                        self.posicoes_itens_por_camada[self.camada_atual] = itens_na_roi.copy()
                        self.logger.info(f"Posições da camada {self.camada_atual} armazenadas: {len(itens_na_roi)} itens")
                    
                    self.contagens_por_camada[self.camada_atual] = self.contagem_estabilizada
                    self.camada_atual += 1
                    self.logger.info(f"Iniciando contagem para camada {self.camada_atual}")
                    self.buffer_contagem_itens.clear()
                    self._transitar_para(ESTADOS['CONTANDO_ITENS'], f"Avançando para camada {self.camada_atual}")
                    return
                
                if self.usar_memoria_espacial and self.camada_atual > 1:
                    # Usar memória espacial para validar se os itens são realmente novos
                    itens_novos = self._verificar_itens_novos(itens_na_roi)
                    percentual_novos = len(itens_novos) / len(itens_na_roi) if itens_na_roi else 0
                    
                    self.logger.info(f"Análise espacial - Itens novos: {len(itens_novos)}/{len(itens_na_roi)} ({percentual_novos:.1%})")
                    
                    if percentual_novos >= PERCENTUAL_ITENS_NOVOS_MINIMO:
                        # Maioria dos itens são novos, pode ser uma camada válida mesmo sem divisor
                        self.logger.warning(f"Divisor ausente, mas {percentual_novos:.1%} dos itens são novos. Considerando camada válida.")
                        
                        # Armazenar posições e avançar
                        self.posicoes_itens_por_camada[self.camada_atual] = itens_na_roi.copy()
                        self.contagens_por_camada[self.camada_atual] = self.contagem_estabilizada
                        
                        if self.camada_atual < PERFIL_CAIXA['total_camadas']:
                            self.camada_atual += 1
                            self.logger.info(f"Avançando para camada {self.camada_atual} (validação espacial)")
                            self.buffer_contagem_itens.clear()
                            self._transitar_para(ESTADOS['CONTANDO_ITENS'], f"Validação espacial - camada {self.camada_atual}")
                        else:
                            total_itens = sum(self.contagens_por_camada.values())
                            self.logger.info(f"Caixa completa por validação espacial! Total: {total_itens} itens")
                            self._transitar_para(ESTADOS['CAIXA_COMPLETA'], "Completa por validação espacial")
                        return
                    else:
                        self.logger.warning(f"Falso positivo confirmado: apenas {percentual_novos:.1%} dos itens são novos")
                else:
                    self.logger.warning(f"Verificação falhou. Divisor ausente na camada {self.camada_atual}. Falso positivo detectado")
                
                self._transitar_para(ESTADOS['ALERTA_DIVISOR_AUSENTE'], "Falso positivo detectado")

        elif estado_atual == ESTADOS['AGUARDANDO_DIVISOR']:
            # Estado onde aguardamos o divisor após completar a camada 1
            if not roi_estavel:
                # Alerta específico: caixa removida após completar camada mas antes do divisor
                if self._pode_alertar("caixa_pos_camada_completa", 5.0):
                    self.logger.error(f"🚨 ALERTA: Caixa removida após completar camada {self.camada_atual-1}!")
                    self.logger.error(f"⚠️  Camada {self.camada_atual-1} estava completa ({PERFIL_CAIXA['itens_por_camada']} itens), aguardando divisor")
                self._transitar_para(ESTADOS['AGUARDANDO_CAIXA'], "ROI perdida aguardando divisor")
                return
            
            if divisor_estavel:
                # Verificar se o divisor já cobriu todos os itens (contagem deve ir a zero)
                contagem_atual = len(itens_na_roi)
                
                if contagem_atual == 0:
                    # Divisor cobrindo todos os itens! Agora pode avançar
                    if not self.divisor_cobrindo_itens:
                        self.logger.info(f"Divisor cobrindo todos os itens da camada {self.camada_atual}. Preparando para avançar.")
                        self.divisor_cobrindo_itens = True
                        
                        # Posições já foram armazenadas quando a camada foi completada
                        
                        self.contagens_por_camada[self.camada_atual] = self.contagem_estabilizada
                        self.camada_atual += 1
                        self.logger.info(f"Iniciando contagem para camada {self.camada_atual} (após divisor cobrir itens)")
                        self.buffer_contagem_itens.clear()
                        self.divisor_cobrindo_itens = False  # Reset para próxima camada
                        self._transitar_para(ESTADOS['CONTANDO_ITENS'], f"Avançando para camada {self.camada_atual}")
                else:
                    # Divisor presente mas ainda não cobriu todos os itens
                    self.logger.debug(f"Divisor presente, mas ainda vendo {contagem_atual} itens. Aguardando cobertura completa.")
                    self.divisor_cobrindo_itens = False
            else:
                # Ainda aguardando divisor - manter estado
                self.logger.debug(f"Aguardando divisor após completar camada {self.camada_atual}")

        elif estado_atual == ESTADOS['ALERTA_DIVISOR_AUSENTE']:
            if divisor_estavel:
                self.logger.info(f"Divisor detectado. Retomando contagem da camada {self.camada_atual}")
                self._transitar_para(ESTADOS['CONTANDO_ITENS'], "Divisor detectado")
            elif not roi_estavel:
                self.logger.warning("Caixa removida durante o alerta de divisor")
                self._transitar_para(ESTADOS['CAIXA_AUSENTE'], "ROI perdida durante alerta")
                self.caixa_ausente_desde = time.time()

        elif estado_atual == ESTADOS['CAIXA_COMPLETA']:
            # O sistema aguarda a caixa ser removida para reiniciar o ciclo.
            if not roi_estavel:
                self.logger.info("Caixa completa removida. Reiniciando o ciclo para a próxima caixa")
                self._resetar_sistema()

        elif estado_atual == ESTADOS['CAIXA_AUSENTE']:
            if roi_estavel:
                tempo_ausente = time.time() - self.caixa_ausente_desde if self.caixa_ausente_desde else 0
                self.logger.info(f"Caixa reapareceu após {tempo_ausente:.1f}s. Retomando contagem")
                self._transitar_para(self.estado_anterior, "ROI reapareceu") # Volta para o estado que estava antes da ausência
                self.caixa_ausente_desde = None
            
            elif self.caixa_ausente_desde and (time.time() - self.caixa_ausente_desde > TEMPO_LIMITE_CAIXA_AUSENTE):
                # Alerta detalhado sobre progresso perdido
                total_itens_perdidos = sum(self.contagens_por_camada.values()) + self.contagem_estabilizada
                self.logger.error(f"🚨 TIMEOUT: Caixa ausente por {TEMPO_LIMITE_CAIXA_AUSENTE}s - RESETANDO SISTEMA")
                self.logger.error(f"📋 PROGRESSO PERDIDO:")
                self.logger.error(f"   - Camada atual: {self.camada_atual}")
                self.logger.error(f"   - Itens na camada atual: {self.contagem_estabilizada}/{PERFIL_CAIXA['itens_por_camada']}")
                self.logger.error(f"   - Total de itens perdidos: {total_itens_perdidos}")
                for camada, contagem in self.contagens_por_camada.items():
                    if contagem > 0:
                        self.logger.error(f"   - Camada {camada}: {contagem} itens")
                self._resetar_sistema()

    def _resetar_sistema(self):
        """Reseta o estado do sistema para o inicial."""
        self.logger.info("Sistema resetado - reiniciando ciclo completo")
        self._transitar_para(ESTADOS['AGUARDANDO_CAIXA'], "Reset do sistema")
        self.camada_atual = 1
        self.contagens_por_camada = {i: 0 for i in range(1, PERFIL_CAIXA['total_camadas'] + 1)}
        self.contagem_estabilizada = 0
        self.caixa_ausente_desde = None
        self.ultima_roi_conhecida = None
        # Limpa a memória espacial
        self.posicoes_itens_por_camada.clear()
        # Limpa os buffers
        self.buffer_roi = []
        self.buffer_contagem_itens = []
        self.buffer_divisor_presente = []
        
        # Controle de estabilização
        self.divisor_detectado_frames = 0
        self.frames_minimos_divisor = 3  # Aguardar 3 frames após detectar divisor
        
        # Flag para detectar início com caixa já cheia
        self.primeira_deteccao = True
        
        # Reset de controles avançados da camada 2
        self.camada_2_estabelecida = False
        self.tempo_ultimo_divisor_ausente = None
        self.tempo_ultima_contagem_baixa = None
        self.divisor_cobrindo_itens = False
        
        # Reset de controles de detecção de saltos híbrida
        self.contagem_anterior_camada_2 = 0
        self.tempo_ultima_contagem_camada_2 = None
        self._reset_controles_salto()
        
        # Reset de controles de carência para perda de caixa
        self.tempo_perda_caixa = None
        self.estado_antes_perda_caixa = None
        
        # Reset de controles de debounce para alertas
        self.ultimo_alerta_tempo = None
        self.ultimo_alerta_tipo = None

    def _pode_alertar(self, tipo_alerta, intervalo_minimo=3.0):
        """Verifica se pode emitir um alerta baseado no debounce."""
        tempo_atual = time.time()
        
        if (self.ultimo_alerta_tipo == tipo_alerta and 
            self.ultimo_alerta_tempo and 
            (tempo_atual - self.ultimo_alerta_tempo) < intervalo_minimo):
            return False
        
        self.ultimo_alerta_tempo = tempo_atual
        self.ultimo_alerta_tipo = tipo_alerta
        return True
    
    def _transitar_para(self, novo_estado, motivo=""):
        """Método auxiliar para registrar a mudança de estado."""
        if self.status_sistema != novo_estado:
            self.estado_anterior = self.status_sistema
            self.logger.info(f"TRANSIÇÃO DE ESTADO: {self.status_sistema} → {novo_estado} - {motivo}")
            self.status_sistema = novo_estado

    def _verificar_itens_novos(self, itens_atuais):
        """
        Verifica quais itens da lista atual são realmente novos comparando
        com as posições dos itens das camadas anteriores.
        
        Args:
            itens_atuais: Lista de coordenadas dos itens detectados atualmente
            
        Returns:
            Lista de itens que são considerados "novos" (não presentes nas camadas anteriores)
        """
        if not self.usar_memoria_espacial or not itens_atuais:
            return itens_atuais
        
        itens_novos = []
        
        for item_atual in itens_atuais:
            eh_novo = True
            x1_atual, y1_atual, x2_atual, y2_atual = item_atual
            centro_atual = ((x1_atual + x2_atual) / 2, (y1_atual + y2_atual) / 2)
            
            # Verificar contra todas as camadas anteriores
            for camada_anterior in range(1, self.camada_atual):
                if camada_anterior in self.posicoes_itens_por_camada:
                    itens_camada_anterior = self.posicoes_itens_por_camada[camada_anterior]
                    
                    for item_anterior in itens_camada_anterior:
                        x1_ant, y1_ant, x2_ant, y2_ant = item_anterior
                        centro_anterior = ((x1_ant + x2_ant) / 2, (y1_ant + y2_ant) / 2)
                        
                        # Calcular distância euclidiana entre os centros
                        distancia = math.sqrt(
                            (centro_atual[0] - centro_anterior[0]) ** 2 + 
                            (centro_atual[1] - centro_anterior[1]) ** 2
                        )
                        
                        if distancia < DISTANCIA_MINIMA_ITEM_NOVO:
                            # Item muito próximo de um item anterior, não é novo
                            eh_novo = False
                            self.logger.debug(f"Item descartado (distância {distancia:.1f}px da camada {camada_anterior})")
                            break
                    
                    if not eh_novo:
                        break
            
            if eh_novo:
                itens_novos.append(item_atual)
                self.logger.debug(f"Item novo confirmado: centro {centro_atual}")
        
        self.logger.info(f"Memória espacial: {len(itens_novos)}/{len(itens_atuais)} itens são novos")
        return itens_novos
    
    def _processar_deteccao_saltos_hibrida(self, contagem_atual, itens_na_roi):
        """
        Lógica híbrida para detectar e validar saltos anômalos na contagem da camada 2:
        1. Detecta saltos suspeitos (>3 itens em <2s)
        2. Valida usando memória espacial (70%+ itens novos)
        3. Aplica carência temporal (3s) para confirmação
        4. Aceita apenas se todas as validações passarem
        
        Retorna True se a contagem é válida, False se deve pausar processamento
        """
        tempo_atual = time.time()
        
        # Só aplicar para camada 2
        if self.camada_atual != 2:
            return True
        
        # Primeira contagem da camada 2, inicializar controles
        if self.tempo_ultima_contagem_camada_2 is None:
            self.contagem_anterior_camada_2 = contagem_atual
            self.tempo_ultima_contagem_camada_2 = tempo_atual
            return True
        
        # Calcular salto e tempo decorrido
        salto = contagem_atual - self.contagem_anterior_camada_2
        tempo_decorrido = tempo_atual - self.tempo_ultima_contagem_camada_2
        
        # 1. DETECÇÃO DE SALTO SUSPEITO COM TOLERÂNCIA A OCLUSÕES
        if salto > SALTO_SUSPEITO_MINIMO and tempo_decorrido < TEMPO_MAXIMO_SALTO:
            # Verificar se é uma oclusão natural (salto pequeno) ou suspeito (salto grande)
            if TOLERANCIA_OCLUSAO_CAMADA_2 and salto <= SALTO_OCLUSAO_MAXIMO:
                # Salto pequeno - provavelmente oclusão natural, aceitar
                self.logger.debug(f"Salto pequeno tolerado (oclusão): {self.contagem_anterior_camada_2} → {contagem_atual} em {tempo_decorrido:.1f}s")
                self.contagem_anterior_camada_2 = contagem_atual
                self.tempo_ultima_contagem_camada_2 = tempo_atual
                return True
            
            # Salto grande - suspeito, aplicar validação
            if not self.salto_suspeito_detectado:
                self.logger.warning(f"Salto suspeito detectado: {self.contagem_anterior_camada_2} → {contagem_atual} em {tempo_decorrido:.1f}s")
                self.salto_suspeito_detectado = True
                self.tempo_inicio_salto_suspeito = tempo_atual
                self.itens_salto_suspeito = itens_na_roi.copy()
                return False  # Pausar processamento
        
        # 2. PROCESSAMENTO DE SALTO SUSPEITO EM VALIDAÇÃO
        if self.salto_suspeito_detectado:
            tempo_carencia = tempo_atual - self.tempo_inicio_salto_suspeito
            
            if tempo_carencia < TEMPO_CARENCIA_SALTO:
                # Ainda em carência, aguardar
                self.logger.debug(f"Salto em validação: {tempo_carencia:.1f}/{TEMPO_CARENCIA_SALTO}s")
                return False
            
            # Carência completa, fazer validação final
            self.logger.info("Carência completa. Validando salto com memória espacial...")
            
            # 3. VALIDAÇÃO ESPACIAL
            if self.usar_memoria_espacial:
                itens_novos = self._verificar_itens_novos(self.itens_salto_suspeito)
                percentual_novos = len(itens_novos) / len(self.itens_salto_suspeito) if self.itens_salto_suspeito else 0
                
                if percentual_novos >= PERCENTUAL_ITENS_NOVOS_SALTO:
                    # SALTO VÁLIDO - Aceitar
                    self.logger.info(f"Salto validado: {percentual_novos:.1%} dos itens são novos. Aceitando contagem.")
                    self._reset_controles_salto()
                    self.contagem_anterior_camada_2 = contagem_atual
                    self.tempo_ultima_contagem_camada_2 = tempo_atual
                    return True
                else:
                    # SALTO INVÁLIDO - Rejeitar e voltar para aguardar divisor
                    self.logger.warning(f"Salto rejeitado: apenas {percentual_novos:.1%} dos itens são novos. Voltando para aguardar divisor.")
                    self._reset_controles_salto()
                    self._voltar_para_aguardar_divisor()
                    return False
            else:
                # Sem memória espacial, aceitar após carência
                self.logger.info("Salto aceito após carência (memória espacial desativada).")
                self._reset_controles_salto()
                self.contagem_anterior_camada_2 = contagem_atual
                self.tempo_ultima_contagem_camada_2 = tempo_atual
                return True
        
        # 4. CONTAGEM NORMAL - Atualizar controles
        self.contagem_anterior_camada_2 = contagem_atual
        self.tempo_ultima_contagem_camada_2 = tempo_atual
        return True
    
    def _reset_controles_salto(self):
        """Reset dos controles de detecção de saltos"""
        self.salto_suspeito_detectado = False
        self.tempo_inicio_salto_suspeito = None
        self.itens_salto_suspeito = []
    
    def _voltar_para_aguardar_divisor(self):
        """Volta para aguardar divisor quando salto é rejeitado"""
        self.logger.info("Retornando para aguardar divisor devido a salto inválido")
        self.camada_atual = 1
        self.camada_2_estabelecida = False
        self.tempo_ultimo_divisor_ausente = None
        self.tempo_ultima_contagem_baixa = None
        self.contagem_anterior_camada_2 = 0
        self.tempo_ultima_contagem_camada_2 = None
        self.buffer_contagem_itens.clear()
        self._transitar_para(ESTADOS['AGUARDANDO_DIVISOR'], "Salto rejeitado - aguardando divisor")
    
    def _processar_logica_camada_2(self, divisor_estavel, contagem_atual):
        """
        Lógica avançada para controle da camada 2:
        - Até 4 itens: Exige divisor presente
        - 5+ itens: Considera camada estabelecida, divisor pode ser ocultado
        - < 5 itens após estabelecida: Volta a exigir divisor com carência
        """
        tempo_atual = time.time()
        
        # Se a camada 2 ainda não foi estabelecida (< 5 itens)
        if not self.camada_2_estabelecida:
            if contagem_atual >= ITENS_MINIMOS_CAMADA_2_ESTABELECIDA:
                # Camada 2 agora está estabelecida!
                self.camada_2_estabelecida = True
                self.logger.info(f"Camada 2 estabelecida com {contagem_atual} itens. Divisor não é mais obrigatório.")
                self.tempo_ultimo_divisor_ausente = None  # Reset carência
                return
            
            # Ainda não estabelecida, exige divisor
            if not divisor_estavel:
                if self.tempo_ultimo_divisor_ausente is None:
                    self.tempo_ultimo_divisor_ausente = tempo_atual
                    self.logger.debug(f"Divisor ausente na camada 2 (não estabelecida). Iniciando carência...")
                    return
                
                tempo_carencia = tempo_atual - self.tempo_ultimo_divisor_ausente
                if tempo_carencia >= TEMPO_CARENCIA_DIVISOR_AUSENTE:
                    self.logger.warning(f"Divisor ausente na camada {self.camada_atual} por {tempo_carencia:.1f}s. Voltando para camada 1.")
                    self._voltar_para_camada_1()
                    return
                else:
                    self.logger.debug(f"Carência divisor ausente: {tempo_carencia:.1f}/{TEMPO_CARENCIA_DIVISOR_AUSENTE}s")
            else:
                # Divisor presente, reset carência
                self.tempo_ultimo_divisor_ausente = None
        
        else:
            # Camada 2 já estabelecida
            if contagem_atual < ITENS_MINIMOS_CAMADA_2_ESTABELECIDA:
                # Contagem baixou, pode ter sido removido itens
                if self.tempo_ultima_contagem_baixa is None:
                    self.tempo_ultima_contagem_baixa = tempo_atual
                    self.logger.debug(f"Contagem baixa na camada 2 estabelecida ({contagem_atual}). Iniciando carência...")
                    return
                
                tempo_carencia = tempo_atual - self.tempo_ultima_contagem_baixa
                if tempo_carencia >= TEMPO_CARENCIA_CONTAGEM_BAIXA:
                    self.logger.warning(f"Contagem baixa por {tempo_carencia:.1f}s. Camada 2 não mais estabelecida.")
                    self.camada_2_estabelecida = False
                    self.tempo_ultima_contagem_baixa = None
                    # Agora volta a exigir divisor
                    if not divisor_estavel:
                        self.logger.warning(f"Divisor também ausente. Voltando para camada 1.")
                        self._voltar_para_camada_1()
                        return
                else:
                    self.logger.debug(f"Carência contagem baixa: {tempo_carencia:.1f}/{TEMPO_CARENCIA_CONTAGEM_BAIXA}s")
            else:
                # Contagem OK, reset carência
                self.tempo_ultima_contagem_baixa = None
    
    def _voltar_para_camada_1(self):
        """Volta para a camada 1 quando há problemas na camada 2"""
        self.logger.info("Retornando para camada 1 devido a problemas na camada 2")
        self.camada_atual = 1
        self.camada_2_estabelecida = False
        self.tempo_ultimo_divisor_ausente = None
        self.tempo_ultima_contagem_baixa = None
        self.buffer_contagem_itens.clear()
        self._transitar_para(ESTADOS['CONTANDO_ITENS'], "Retorno para camada 1")
    
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
