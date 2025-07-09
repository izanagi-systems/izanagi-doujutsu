import cv2
import os
import datetime

def capturar_imagens():
    """
    Abre a webcam para capturar imagens para o dataset.
    - Pressione a BARRA DE ESPAÇO para salvar o frame atual.
    - Pressione 'q' para sair.
    """
    # --- Configuração ---
    ID_CAMERA = 0
    PASTA_SAIDA = os.path.join('dataset', 'images')
    # --------------------

    # Cria a pasta de saída se ela não existir
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    print(f"[INFO] Imagens serão salvas em: {os.path.abspath(PASTA_SAIDA)}")

    cap = cv2.VideoCapture(ID_CAMERA)
    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a câmera. Verifique o ID da câmera.")
        return

    print("[INFO] Câmera iniciada. Pressione ESPAÇO para capturar ou 'q' para sair.")
    capturas = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha ao capturar o frame.")
            break

        # Mostra o feed da câmera ao vivo
        cv2.imshow("Capturador de Imagens - Pressione ESPAÇO", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Encerrando o capturador.")
            break
        elif key == ord(' '): # Barra de espaço
            # Gera um nome de arquivo único com base no timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            nome_arquivo = f"capture_{timestamp}.jpg"
            caminho_completo = os.path.join(PASTA_SAIDA, nome_arquivo)
            
            # Salva o frame
            cv2.imwrite(caminho_completo, frame)
            capturas += 1
            print(f"[CAPTURA {capturas}] Imagem salva: {nome_arquivo}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Sessão finalizada. Total de {capturas} imagens capturadas.")

if __name__ == "__main__":
    capturar_imagens()
