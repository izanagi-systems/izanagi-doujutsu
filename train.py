import shutil
from ultralytics import YOLO
import argparse
import os
from utils.s3 import S3



def treinar_modelo(object_type, epochs, imgsz, run_name):
    """
    Carrega um modelo YOLO pré-treinado e inicia o treinamento
    com o dataset customizado especificado.

    :param data_path: Caminho completo para o arquivo data.yaml.
    :param epochs: Número de épocas para o treinamento.
    :param imgsz: Tamanho da imagem para o treinamento.
    :param run_name: Nome específico para esta execução (run) dentro do projeto.
    """

    S3().download_data(local_dir='./tmp', prefix=object_type)
    
    data_path = f'./tmp/{object_type}/data.yaml'

    if not os.path.exists(data_path):
        print(f"[ERRO] Arquivo de configuração não encontrado em: {data_path}")
        return

    # Carrega um modelo pré-treinado. 'yolov8n.pt' é o menor e mais rápido.
    model = YOLO('yolov8n.pt')

    # Inicia o treinamento com os parâmetros recebidos
    print(f"-- Iniciando treinamento para o dataset: {data_path} --")
    print(f"-- Configuração: {epochs} épocas, tamanho da imagem {imgsz} --")
    print(f"-- Salvando em: runs/detect/{run_name} --")
    results = model.train(
        data=data_path, 
        epochs=epochs, 
        imgsz=imgsz,
        project='runs/detect', # Força o salvamento na pasta correta
        name=run_name
    )

    print("\n-- Treinamento Concluído --")
    print(f"Resultados salvos em: {results.save_dir}")

    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
        print("[INFO] Pasta './tmp' removida.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para treinar um modelo YOLOv8.")
    parser.add_argument('--object_type', type=str, required=True, help="Tipo de objeto: 'roi detector' ou 'items'.")
    parser.add_argument('--epochs', type=int, default=100, help="Número de épocas para o treinamento.")
    parser.add_argument('--imgsz', type=int, default=640, help="Tamanho da imagem (altura e largura) para o treinamento.")
    parser.add_argument('--name', type=str, default='train', help="Nome da execução específica (run) que será salva dentro de 'runs/detect'.")

    args = parser.parse_args()

    treinar_modelo(
        object_type=args.object_type,
        epochs=args.epochs, 
        imgsz=args.imgsz,
        run_name=args.name
    )