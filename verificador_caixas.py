import cv2
from ultralytics import YOLO

class ContadorDeItens:
    def __init__(self, video_source=0, model_path='runs/detect/contador_itens_aug2/weights/best.pt'):
        """
        Inicializa o contador de itens.
        :param video_source: ID da câmera (ex: 0) ou caminho para o arquivo de vídeo.
        :param model_path: Caminho para o modelo YOLO treinado (.pt).
        """
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise IOError(f"Não foi possível acessar a fonte de vídeo: {video_source}.")
        
        # Carrega o modelo YOLO treinado
        print(f"Carregando modelo de: {model_path}")
        self.model = YOLO(model_path)
        # Verifica se a fonte é uma câmera ao vivo (int) ou um arquivo de vídeo (str)
        self.is_live = isinstance(video_source, int)

    def _loop_video_file(self):
        """
        Loop de processamento para arquivos de vídeo com controles de player.
        Espaço: Play/Pause. 'd'/'a': Frame a frame (quando pausado). 'q': Sair.
        """
        paused = True
        ret, frame = self.cap.read()

        while True:
            if ret: # Só processa e exibe se o frame for válido
                results = self.model(frame, verbose=False)
                result = results[0]
                frame_vis = result.plot()
                contagem = len(result.boxes)
                cv2.putText(frame_vis, f"Contagem: {contagem}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow("Contador de Itens - YOLO", frame_vis)
            else:
                print("[INFO] Fim do vídeo. Pressione 'q' para sair ou 'a' para reiniciar.")
                paused = True # Força a pausa no final

            # Lógica de controle de vídeo
            if paused:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('d') and paused:
                ret, frame = self.cap.read()
            elif key == ord('a'):
                pos_atual = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                pos_anterior = max(0, pos_atual - 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos_anterior)
                ret, frame = self.cap.read()
            
            if not paused:
                ret, frame = self.cap.read()

    def _loop_live_camera(self):
        """
        Loop de processamento para câmera ao vivo.
        Pressione 'q' para sair.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERRO] Não foi possível ler o frame da câmera. Encerrando.")
                break

            results = self.model(frame, verbose=False)
            result = results[0]
            frame_vis = result.plot()
            contagem = len(result.boxes)
            cv2.putText(frame_vis, f"Contagem: {contagem}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("Contador de Itens - YOLO", frame_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        """
        Inicia o processamento, escolhendo o loop apropriado (vídeo ou câmera).
        """
        if self.is_live:
            print("[INFO] Iniciando modo Câmera ao Vivo. Pressione 'q' para sair.")
            self._loop_live_camera()
        else:
            print("[INFO] Iniciando modo Arquivo de Vídeo.")
            self._loop_video_file()
        
        self.cap.release()
        cv2.destroyAllWindows()


# Execução do sistema
if __name__ == "__main__":
    # --- CONFIGURAÇÃO --- #
    USAR_CAMERA = False  # Mude para True para usar a webcam, False para usar um arquivo de vídeo.
    FONTE_CAMERA = 0     # ID da câmera (geralmente 0 para a webcam principal).
    
    FONTE_VIDEO = r"c:\Users\ti-005\Desktop\pvcf_gpt\videos_test\WhatsApp Video 2025-07-08 at 14.06.58.mp4"
    CAMINHO_MODELO = 'runs/detect/contador_itens_aug2/weights/best.pt'
    # -------------------- #

    fonte_final = FONTE_CAMERA if USAR_CAMERA else FONTE_VIDEO
    
    sistema = ContadorDeItens(video_source=fonte_final, model_path=CAMINHO_MODELO)
    
    try:
        sistema.run()
    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro inesperado: {e}")
