import cv2
import numpy as np
import os
import time

# Desabilita o sync da ultralytics para evitar downloads
os.environ['ULTRALYTICS_SYNC'] = 'False'

# Importa todas as configurações e constantes
from config import *
from detector import Detector
from state_manager import StateManager
from visualizer import Visualizer
from logger_config import init_siac_logging, get_siac_logger, SiacLogger

class SiacApp:
    """Classe principal que orquestra o sistema SIAC."""
    def __init__(self):
        # Inicializar sistema de logging
        init_siac_logging(log_level="INFO", enable_file_logging=True)
        self.logger = get_siac_logger("SIAC_APP")
        
        self.logger.info("Iniciando carregamento dos módulos do sistema SIAC")
        
        try:
            self.detector = Detector()
            self.state_manager = StateManager()
            self.visualizer = Visualizer()
            
            # Métricas de performance
            self.fps_counter = 0
            self.last_fps_time = time.time()
            self.current_fps = 0.0
            
            self.logger.info("Todos os módulos carregados com sucesso")
            
        except Exception as e:
            SiacLogger.log_error_with_context(self.logger, e, "Inicialização dos módulos")
            raise

    def run(self, video_source=0):
        self.logger.info(f"Tentando abrir fonte de vídeo: {video_source}")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error(f"Falha ao abrir a fonte de vídeo: {video_source}")
            return

        self.logger.info("Fonte de vídeo aberta com sucesso. Iniciando processamento...")
        self.logger.info("Pressione 'q' para encerrar o sistema")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Falha ao capturar frame ou fim do vídeo")
                    break

                start_time = time.time()
                frame_processado = self.processar_frame(frame)
                processing_time = (time.time() - start_time) * 1000  # em ms
                
                # Calcular FPS
                frame_count += 1
                self._update_fps_metrics()
                
                # Log de performance a cada 60 frames (reduzido para menos spam)
                if frame_count % 60 == 0:
                    SiacLogger.log_performance_metrics(self.logger, self.current_fps, processing_time)
                
                cv2.imshow('SIAC - Verificador de Caixas', frame_processado)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Comando de saída recebido pelo usuário")
                    break
                    
        except Exception as e:
            SiacLogger.log_error_with_context(self.logger, e, "Loop principal de processamento")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info(f"Processamento finalizado. Total de frames processados: {frame_count}")

    def processar_frame(self, frame):
        frame_desenhado = frame.copy()

        try:
            # 1. Realizar detecção de todos os objetos
            resultados = self.detector.detectar_objetos(frame)
            todos_itens = resultados['itens']
            todos_divisores = resultados['divisores']
            rois_detectadas = resultados['caixas']
            divisores_baixa_confianca = resultados.get('divisores_baixa_confianca', [])

            # Log de detecções (apenas em modo DEBUG)
            SiacLogger.log_detection_stats(
                self.logger, 
                len(rois_detectadas), 
                len(todos_itens), 
                len(todos_divisores)
            )

            # 2. Encontrar a ROI de maior área para ser a ROI ativa
            roi_ativa = self._get_roi_maior_area(rois_detectadas)

            # 3. Filtrar objetos que estão dentro da ROI ativa
            itens_na_roi = []
            divisores_na_roi = []
            if roi_ativa:
                itens_na_roi = self._filtrar_objetos_na_roi(todos_itens, roi_ativa)
                divisores_na_roi = self._filtrar_objetos_na_roi(todos_divisores, roi_ativa)

            # 4. Atualizar a máquina de estados com as detecções atuais
            self.state_manager.atualizar_estado(roi_ativa, itens_na_roi, divisores_na_roi)

            # 5. Obter o status REAL do sistema para a visualização
            status_visual = self.state_manager.get_status_visual()

            # 6. Desenhar as visualizações usando o Visualizer
            self.visualizer.desenhar_visualizacoes(
                frame_desenhado, 
                roi_ativa, 
                itens_na_roi, 
                divisores_na_roi, 
                status_visual
            )
            
        except Exception as e:
            SiacLogger.log_error_with_context(self.logger, e, "Processamento do frame")
            # Em caso de erro, retorna o frame original
            frame_desenhado = frame.copy()

        return frame_desenhado

    def _get_roi_maior_area(self, rois):
        """De uma lista de ROIs, retorna a que tiver a maior área."""
        if not rois:
            return None
        
        areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rois]
        maior_roi_idx = np.argmax(areas)
        maior_area = areas[maior_roi_idx]
        
        # Log de área removido para evitar spam
        return rois[maior_roi_idx]

    def _filtrar_objetos_na_roi(self, objetos, roi):
        """Filtra uma lista de objetos, retornando apenas os que estão dentro da ROI."""
        objetos_filtrados = []
        rx1, ry1, rx2, ry2 = roi
        for obj in objetos:
            ox1, oy1, ox2, oy2 = obj
            # Verifica se o centro do objeto está dentro da ROI
            centro_x, centro_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            if rx1 < centro_x < rx2 and ry1 < centro_y < ry2:
                objetos_filtrados.append(obj)
        return objetos_filtrados

    def _update_fps_metrics(self):
        """Atualiza as métricas de FPS."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Atualiza a cada segundo
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time

if __name__ == '__main__':
    try:
        app = SiacApp()
        app.run(video_source=0)  # Use 0 para webcam ou 'caminho/para/video.mp4'
    except KeyboardInterrupt:
        print("\nSistema interrompido pelo usuário")
    except Exception as e:
        print(f"Erro fatal no sistema: {e}")