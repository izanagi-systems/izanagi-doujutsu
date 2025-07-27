from ultralytics import YOLO
from config import MODELOS, CONFIDENCIA_LIMITE, CONFIDENCIA_DIVISOR, DEBUG_DIVISORES, DEBUG_DIVISORES_VERBOSE
from logger_config import get_siac_logger, SiacLogger
import os
import time

class Detector:
    """
    Encapsula a lógica de detecção de objetos com os modelos YOLO.
    """
    def __init__(self):
        """
        Carrega os modelos de detecção de ROI e de itens.
        """
        self.logger = get_siac_logger("DETECTOR")
        
        # Controle de logs para evitar spam
        self.ultimo_log_divisores = 0
        self.intervalo_log_divisores = 5  # Log a cada 5 segundos
        self.ultimo_status_divisores = None
        
        self.logger.info("Iniciando carregamento dos modelos de detecção")
        
        try:
            # Verificar se os arquivos de modelo existem
            roi_model_path = MODELOS['roi_detector']
            item_model_path = MODELOS['item_detector']
            
            if not os.path.exists(roi_model_path):
                raise FileNotFoundError(f"Modelo ROI não encontrado: {roi_model_path}")
            if not os.path.exists(item_model_path):
                raise FileNotFoundError(f"Modelo de itens não encontrado: {item_model_path}")
            
            self.logger.info(f"Carregando modelo ROI: {roi_model_path}")
            self.roi_model = YOLO(roi_model_path)
            
            self.logger.info(f"Carregando modelo de itens: {item_model_path}")
            self.item_model = YOLO(item_model_path)
            
            self.logger.info("Todos os modelos de detecção carregados com sucesso")
            self.logger.info(f"Confiança mínima configurada: {CONFIDENCIA_LIMITE}")
            self.logger.info(f"Confiança para divisores: {CONFIDENCIA_DIVISOR}")
            self.logger.info(f"Debug de divisores: {'Ativado' if DEBUG_DIVISORES else 'Desativado'}")
            
        except Exception as e:
            SiacLogger.log_error_with_context(self.logger, e, "Carregamento dos modelos")
            raise

    def detectar_objetos(self, frame):
        """
        Executa a detecção de ROI (caixa) e de itens/divisores no frame.

        Args:
            frame: O frame do vídeo a ser processado.

        Returns:
            Um dicionário contendo as listas de bounding boxes para cada classe:
            {'caixas': [], 'itens': [], 'divisores': []}
        """
        try:
            # 1. Detectar a ROI (caixas)
            self.logger.debug("Executando detecção de ROI")
            deteccoes_roi = self.roi_model.predict(source=frame, conf=CONFIDENCIA_LIMITE, verbose=False)[0]
            caixas_detectadas = [list(map(int, box.xyxy[0].tolist())) for box in deteccoes_roi.boxes]

            # 2. Detectar Itens e Divisores com configurações específicas
            self.logger.debug("Executando detecção de itens e divisores")
            
            # Detectar com confiança padrão primeiro
            deteccoes_itens = self.item_model.predict(source=frame, conf=CONFIDENCIA_LIMITE, verbose=False)[0]
            # Detectar divisores com confiança mais baixa
            deteccoes_divisores = self.item_model.predict(source=frame, conf=CONFIDENCIA_DIVISOR, verbose=False)[0]
            
            itens_detectados = []
            divisores_detectados = []
            divisores_baixa_confianca = []

            # Processar detecções com confiança padrão
            for box in deteccoes_itens.boxes:
                coords = list(map(int, box.xyxy[0].tolist()))
                confianca = float(box.conf[0])
                
                # Classe 0 é 'item', Classe 1 é 'divisor'
                if int(box.cls) == 0:
                    itens_detectados.append(coords)
                    self.logger.debug(f"Item detectado com confiança {confianca:.2f}: {coords}")
                elif int(box.cls) == 1:
                    divisores_detectados.append(coords)
                    if DEBUG_DIVISORES_VERBOSE:
                        self.logger.info(f"Divisor detectado (alta confiança) {confianca:.2f}: {coords}")
            
            # Processar detecções de divisores com confiança baixa
            for box in deteccoes_divisores.boxes:
                coords = list(map(int, box.xyxy[0].tolist()))
                confianca = float(box.conf[0])
                
                if int(box.cls) == 1 and confianca < CONFIDENCIA_LIMITE:
                    # Divisor com confiança baixa, não adicionado à lista principal
                    divisores_baixa_confianca.append((coords, confianca))
                    if DEBUG_DIVISORES_VERBOSE:
                        self.logger.warning(f"Divisor detectado (baixa confiança) {confianca:.2f}: {coords}")
            
            # Log inteligente sobre divisores (evita spam)
            if DEBUG_DIVISORES:
                current_time = time.time()
                total_divisores_candidatos = len(divisores_detectados) + len(divisores_baixa_confianca)
                status_atual = f"alta:{len(divisores_detectados)},baixa:{len(divisores_baixa_confianca)}"
                
                # Só loga se o status mudou OU se passou tempo suficiente
                if (status_atual != self.ultimo_status_divisores or 
                    current_time - self.ultimo_log_divisores > self.intervalo_log_divisores):
                    
                    if total_divisores_candidatos == 0:
                        self.logger.warning("NENHUM DIVISOR detectado!")
                    else:
                        self.logger.info(f"Divisores: {len(divisores_detectados)} (alta conf.) + {len(divisores_baixa_confianca)} (baixa conf.)")
                    
                    self.ultimo_log_divisores = current_time
                    self.ultimo_status_divisores = status_atual

            # Log do resultado final
            total_deteccoes = len(caixas_detectadas) + len(itens_detectados) + len(divisores_detectados)
            if total_deteccoes > 0:
                self.logger.debug(f"Detecção concluída - ROI: {len(caixas_detectadas)}, Itens: {len(itens_detectados)}, Divisores: {len(divisores_detectados)}")
            
            # Log específico para debug de divisores (apenas se verboso)
            if DEBUG_DIVISORES_VERBOSE and len(divisores_baixa_confianca) > 0:
                self.logger.info(f"Divisores com baixa confiança disponíveis: {len(divisores_baixa_confianca)}")

            return {
                'caixas': caixas_detectadas,
                'itens': itens_detectados,
                'divisores': divisores_detectados,
                'divisores_baixa_confianca': divisores_baixa_confianca  # Para debug
            }
            
        except Exception as e:
            SiacLogger.log_error_with_context(self.logger, e, "Detecção de objetos")
            # Em caso de erro, retorna listas vazias
            return {
                'caixas': [],
                'itens': [],
                'divisores': []
            }